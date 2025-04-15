# proto文件生成

```
./generate_proto.sh
```

# TODO
- [] 多个函数内是否可以公用一个ClientContext, 是否可以提升调用速度
- [] KeepAlive功能是否默认开启


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
