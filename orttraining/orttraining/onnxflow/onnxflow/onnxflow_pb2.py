# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: onnxflow.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='onnxflow.proto',
  package='onnxflow',
  syntax='proto2',
  serialized_options=b'H\003',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0eonnxflow.proto\x12\x08onnxflow\x1a\x19google/protobuf/any.proto\"d\n\x11OnnxFlowParameter\x12\"\n\x04\x64\x61ta\x18\x01 \x02(\x0b\x32\x14.google.protobuf.Any\x12\x15\n\rrequires_grad\x18\x02 \x02(\x08\x12\x14\n\x0cis_parameter\x18\x03 \x02(\x08\"E\n\x12OnnxFlowParameters\x12/\n\nparameters\x18\x01 \x03(\x0b\x32\x1b.onnxflow.OnnxFlowParameterB\x02H\x03'
  ,
  dependencies=[google_dot_protobuf_dot_any__pb2.DESCRIPTOR,])




_ONNXFLOWPARAMETER = _descriptor.Descriptor(
  name='OnnxFlowParameter',
  full_name='onnxflow.OnnxFlowParameter',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='onnxflow.OnnxFlowParameter.data', index=0,
      number=1, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='requires_grad', full_name='onnxflow.OnnxFlowParameter.requires_grad', index=1,
      number=2, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='is_parameter', full_name='onnxflow.OnnxFlowParameter.is_parameter', index=2,
      number=3, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=55,
  serialized_end=155,
)


_ONNXFLOWPARAMETERS = _descriptor.Descriptor(
  name='OnnxFlowParameters',
  full_name='onnxflow.OnnxFlowParameters',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='parameters', full_name='onnxflow.OnnxFlowParameters.parameters', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=157,
  serialized_end=226,
)

_ONNXFLOWPARAMETER.fields_by_name['data'].message_type = google_dot_protobuf_dot_any__pb2._ANY
_ONNXFLOWPARAMETERS.fields_by_name['parameters'].message_type = _ONNXFLOWPARAMETER
DESCRIPTOR.message_types_by_name['OnnxFlowParameter'] = _ONNXFLOWPARAMETER
DESCRIPTOR.message_types_by_name['OnnxFlowParameters'] = _ONNXFLOWPARAMETERS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

OnnxFlowParameter = _reflection.GeneratedProtocolMessageType('OnnxFlowParameter', (_message.Message,), {
  'DESCRIPTOR' : _ONNXFLOWPARAMETER,
  '__module__' : 'onnxflow_pb2'
  # @@protoc_insertion_point(class_scope:onnxflow.OnnxFlowParameter)
  })
_sym_db.RegisterMessage(OnnxFlowParameter)

OnnxFlowParameters = _reflection.GeneratedProtocolMessageType('OnnxFlowParameters', (_message.Message,), {
  'DESCRIPTOR' : _ONNXFLOWPARAMETERS,
  '__module__' : 'onnxflow_pb2'
  # @@protoc_insertion_point(class_scope:onnxflow.OnnxFlowParameters)
  })
_sym_db.RegisterMessage(OnnxFlowParameters)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
