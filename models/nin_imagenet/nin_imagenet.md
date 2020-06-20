# Nin Imagenet Model

## Steps to obtain weights:

- downloaded caffemodel and prototxt file from [gist](https://gist.github.com/mavenlin/d802a5849de39225bcc6)
    - did not work because of the following error `Error encountered: Multiple top nodes are not supported.`
- Thankfully, someone created a cleaner prototxt file [here](https://gist.github.com/tzutalin/0e3fd793a5b13dd7f647) 
- had to update the prototxt file and caffemodel files throught the caffe tools `upgrade_net_proto_binary` and `upgrade_net_proto_text` as described here [here](https://stackoverflow.com/questions/35806105/how-can-i-upgrade-my-caffe-model-so-it-doesnt-upgrade-every-time-i-use-it)
- then converted the files to tensorflow using [**caffe-tensorflow**](https://github.com/ethereon/caffe-tensorflow) library
- (optional) you can skip the previous step and extract the weights directly, using the [**extract-caffe-params**](https://github.com/nilboy/extract-caffe-params) library
- (optional) a useful visualization tool to render caffe networks as images online, [here](http://ethereon.github.io/netscope/quickstart.html)