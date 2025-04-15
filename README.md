# proto文件生成

```
./generate_proto.sh
```


# 参考
**消息类**的头文件和实现文件生成
```
# workspace: ${CMAKE_SOURCE_DIR}
protoc -I proto --cpp_out=. proto/route_guide.proto
```

**服务类**的头文件和实现文件生成
```
# workspace: ${CMAKE_SOURCE_DIR}
protoc -I proto --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/route_guide.proto
```
