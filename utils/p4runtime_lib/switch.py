# Copyright 2017-present Open Networking Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from queue import Queue
from abc import abstractmethod
from datetime import datetime

import grpc
from p4.v1 import p4runtime_pb2
from p4.v1 import p4runtime_pb2_grpc
from p4.tmp import p4config_pb2

MSG_LOG_MAX_LEN = 1024

# List of all active connections
connections = []

def ShutdownAllSwitchConnections():
    for c in connections:
        c.shutdown()

class SwitchConnection(object):

    def __init__(self, name=None, address='127.0.0.1:50051', device_id=0,
                 proto_dump_file=None):
        self.name = name
        self.address = address
        self.device_id = device_id
        self.p4info = None
        self.channel = grpc.insecure_channel(self.address)
        if proto_dump_file is not None:
            interceptor = GrpcRequestLogger(proto_dump_file)
            self.channel = grpc.intercept_channel(self.channel, interceptor)
        self.client_stub = p4runtime_pb2_grpc.P4RuntimeStub(self.channel)
        self.requests_stream = IterableQueue()
        self.stream_msg_resp = self.client_stub.StreamChannel(iter(self.requests_stream))
        self.proto_dump_file = proto_dump_file
        self._reqs = []
        connections.append(self)

    @abstractmethod
    def buildDeviceConfig(self, **kwargs):
        return p4config_pb2.P4DeviceConfig()

    def shutdown(self):
        self.requests_stream.close()
        self.stream_msg_resp.cancel()

    def MasterArbitrationUpdate(self, dry_run=False, **kwargs):
        request = p4runtime_pb2.StreamMessageRequest()
        request.arbitration.device_id = self.device_id
        request.arbitration.election_id.high = 0
        request.arbitration.election_id.low = 1

        if dry_run:
            print("P4Runtime MasterArbitrationUpdate: ", request)
        else:
            self.requests_stream.put(request)
            for item in self.stream_msg_resp:
                return item # just one

    def SetForwardingPipelineConfig(self, p4info, dry_run=False, **kwargs):
        device_config = self.buildDeviceConfig(**kwargs)
        request = p4runtime_pb2.SetForwardingPipelineConfigRequest()
        request.election_id.low = 1
        request.device_id = self.device_id
        config = request.config

        config.p4info.CopyFrom(p4info)
        config.p4_device_config = device_config.SerializeToString()

        request.action = p4runtime_pb2.SetForwardingPipelineConfigRequest.VERIFY_AND_COMMIT
        if dry_run:
            print("P4Runtime SetForwardingPipelineConfig:", request)
        else:
            self.client_stub.SetForwardingPipelineConfig(request)

    def WriteTableEntry(self, table_entry, dry_run=False):
        request = p4runtime_pb2.WriteRequest()
        request.device_id = self.device_id
        request.election_id.low = 1
        update = request.updates.add()
        if table_entry.is_default_action:
            update.type = p4runtime_pb2.Update.MODIFY
        else:
            update.type = p4runtime_pb2.Update.INSERT
        update.entity.table_entry.CopyFrom(table_entry)
        if dry_run:
            print("P4Runtime Write:", request)
        else:
            self.client_stub.Write(request)
            self._reqs.append(request)

    def DeleteTableEntry(self, dry_run=False):
        print("DeleteTableEntry() is called, device_id=", self.device_id)
        updates = []
        for req in reversed(self._reqs):
            for update in reversed(req.updates):
                if update.type == p4runtime_pb2.Update.INSERT:
                    updates.append(update)
        new_req = p4runtime_pb2.WriteRequest()
        new_req.device_id = self.device_id
        new_req.election_id.low = 1
        for update in updates:
            update.type = p4runtime_pb2.Update.DELETE
            new_req.updates.add().CopyFrom(update)
        if dry_run:
            print("P4 Runtime delete:", new_req)
        else:
            self.client_stub.Write(new_req)

    def ReadTableEntries(self, table_id=None, dry_run=False):
        request = p4runtime_pb2.ReadRequest()
        request.device_id = self.device_id
        entity = request.entities.add()
        table_entry = entity.table_entry
        if table_id is not None:
            table_entry.table_id = table_id
        else:
            table_entry.table_id = 0
        if dry_run:
            print("P4Runtime Read:", request)
        else:
            for response in self.client_stub.Read(request):
                yield response

    def ReadCounters(self, counter_id=None, index=None, dry_run=False):
        request = p4runtime_pb2.ReadRequest()
        request.device_id = self.device_id
        entity = request.entities.add()
        counter_entry = entity.counter_entry
        if counter_id is not None:
            counter_entry.counter_id = counter_id
        else:
            counter_entry.counter_id = 0
        if index is not None:
            counter_entry.index.index = index
        if dry_run:
            print("P4Runtime Read:", request)
        else:
            for response in self.client_stub.Read(request):
                yield response

    # Delete
    def ResetCounters(self, counter_id=None, dry_run=False):
        print("ResetCounters() is called, reading from counter_id=", counter_id)
        request = p4runtime_pb2.ReadRequest()
        request.device_id = self.device_id
        entity = request.entities.add()
        counter_entry = entity.counter_entry
        if counter_id is not None:
            counter_entry.counter_id = counter_id
        else:
            counter_entry.counter_id = 0
        updates = []
        for req in reversed(self._reqs):
            for update in reversed(req.updates):
                if update.type == p4runtime_pb2.Update.INSERT:
                    updates.append(update)
        # print("ResetCounters() is called, reading from counter_id=", counter_id)
        new_req = p4runtime_pb2.WriteRequest()
        new_req.device_id = self.device_id
        new_req.election_id.low = 1
        # update = new_req.updates.add()
        # update.type = p4runtime_pb2.Update.DELETE
        # new_req.updates.add().CopyFrom(update)
        # update.entity.counter_entry.CopyFrom(counter_entry)
        for update in updates:
            update.type = p4runtime_pb2.Update.DELETE
            new_req.updates.add().CopyFrom(update)
            update.entity.counter_entry.CopyFrom(counter_entry)
        if dry_run:
            print("ResetCounters() Delete:", new_req)
        else:
            print("new_req", new_req)
            self.client_stub.Write(new_req)

    	# entity = request.entities.add()
    	# counter_entry = entity.counter_entry
    	# request.election_id.low = 1
    	# update = request.updates.add()
    	# # Assign DELETE Type for it
    	# update.type = p4runtime_pb2.Update.DELETE
    	# update.entity.counter_entry.CopyFrom(counter_entry)
    	# if dry_run:
    	# 	print("ResetCounters() Delete: ", request)
    	# else:
    	# 	self.client_stub.Write(request)
        #
        # new_req = p4runtime_pb2.WriteRequest()
        # new_req.device_id = self.device_id
        # new_req.election_id.low = 1
        # for update in updates:
        #     update.type = p4runtime_pb2.Update.DELETE
        #     new_req.updates.add().CopyFrom(update)
        # if dry_run:
        #     print("P4 Runtime delete:", new_req)
        # else:
        #     self.client_stub.Write(new_req)


    def WritePREEntry(self, pre_entry, dry_run=False):
        request = p4runtime_pb2.WriteRequest()
        request.device_id = self.device_id
        request.election_id.low = 1
        update = request.updates.add()
        update.type = p4runtime_pb2.Update.INSERT
        update.entity.packet_replication_engine_entry.CopyFrom(pre_entry)
        if dry_run:
            print("P4Runtime Write:", request)
        else:
            self.client_stub.Write(request)

    def PacketOut(self, packet, dry_run=False, **kwargs):
    	request = p4runtime_pb2.StreamMessageRequest()
    	request.packet.CopyFrom(packet)
    	if dry_run:
    		print("P4 Runtime WritePacketOut: ", request )
    	else:
    		self.requests_stream.put(request)
    		for item in self.stream_msg_resp:
    			return item

    def PacketIn(self, dry_run=False, **kwargs):
        request = p4runtime_pb2.StreamMessageRequest()
        if dry_run:
            print("P4 Runtime PacketIn: ", request)
        else:
            self.requests_stream.put(request)
            for item in self.stream_msg_resp:
                return item

    def WritePacketOut(self, packet, dry_run=False, **kwargs):
    	request = p4runtime_pb2.StreamMessageRequest()
    	request.packet.CopyFrom(packet)
    	if dry_run:
    	    print("P4 Runtime WritePacketOut: ", request)
    	else:
    	    self.requests_stream.put(request)
    	    for item in self.stream_msg_resp:
    		    return item

    def ReadPacketIn(self, dry_run=False, **kwargs):
    	request = p4runtime_pb2.StreamMessageRequest()
    	if dry_run:
    	    print("P4 Runtime ReadPacketIn: ", request)
    	else:
    	    self.requests_stream.put(request)
    	    for item in self.stream_msg_resp:
    		    return item

    # Digest
    def WriteDigestEntry(self, digest_entry, dry_run=False):
        request = p4runtime_pb2.WriteRequest()
        request.device_id = self.device_id
        request.election_id.low = 1
        update = request.updates.add()
        update.type = p4runtime_pb2.Update.INSERT
        update.entity.digest_entry.CopyFrom(digest_entry)

        if dry_run:
            print("P4Runtime write DigestEntry: ", request )
        else:
            self.client_stub.Write(request)

    def DigestListAck(self, digest_ack, dry_run=False, **kwargs):
        request = p4runtime_pb2.StreamMessageRequest()
        request.digest_ack.CopyFrom(digest_ack)
        if dry_run:
            print("P4 Runtime DigestListAck: ", request )
        else:
            self.requests_stream.put(request)
            for item in self.stream_msg_resp:
                return item

    def DigestList(self, dry_run=False, **kwargs):
        request = p4runtime_pb2.StreamMessageRequest()
        if dry_run:
            print("P4 Runtime DigestList Response: ", request )
        else:
            self.requests_stream.put(request)
            for item in self.stream_msg_resp:
                return item


class GrpcRequestLogger(grpc.UnaryUnaryClientInterceptor,
                        grpc.UnaryStreamClientInterceptor):
    """Implementation of a gRPC interceptor that logs request to a file"""

    def __init__(self, log_file):
        self.log_file = log_file
        with open(self.log_file, 'w') as f:
            # Clear content if it exists.
            f.write("")

    def log_message(self, method_name, body):
        with open(self.log_file, 'a') as f:
            ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            msg = str(body)
            f.write("\n[%s] %s\n---\n" % (ts, method_name))
            if len(msg) < MSG_LOG_MAX_LEN:
                f.write(str(body))
            else:
                f.write("Message too long (%d bytes)! Skipping log...\n" % len(msg))
            f.write('---\n')

    def intercept_unary_unary(self, continuation, client_call_details, request):
        self.log_message(client_call_details.method, request)
        return continuation(client_call_details, request)

    def intercept_unary_stream(self, continuation, client_call_details, request):
        self.log_message(client_call_details.method, request)
        return continuation(client_call_details, request)

class IterableQueue(Queue):
    _sentinel = object()

    def __iter__(self):
        return iter(self.get, self._sentinel)

    def close(self):
        self.put(self._sentinel)
