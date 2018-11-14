# Install 
1 add depthwise convolutional layer into darknet
   First, open this file /src/parser.c.
  1) add below into it
        
    #include "utils.h"
    ++ #include "depthwise_convolutional_layer.h" //added by chen
  
  2) find function 'string_to_layer_type' added below into it
    
    if (strcmp(type, "[upsample]")==0) return UPSAMPLE; 
    ++ if (strcmp(type, "[depthwise_convolutional]") == 0) return DEPTHWISE_CONVOLUTIONAL; //added by chen 
         return BLANK;
  
  3) find function 'parse_network_cfg' added below into it
   
    if(lt == CONVOLUTIONAL){ 
        l = parse_convolutional(options, params); 
        } 
     ++ else if (lt == DEPTHWISE_CONVOLUTIONAL) { 
         ++ l = parse_depthwise_convolutional(options, params); //added by chen 
         ++ } 
         else if(lt == DECONVOLUTIONAL){ l = parse_deconvolutional(options, params); 
     }
  
  4) add this function 'parse_depthwise_convolutional' into it

    //added by chen 
    depthwise_convolutional_layer parse_depthwise_convolutional(list *options, size_params params) 
    { 
        int size = option_find_int(options, "size", 1); 
        int stride = option_find_int(options, "stride", 1); 
        int pad = option_find_int_quiet(options, "pad", 0); 
        int padding = option_find_int_quiet(options, "padding", 0); 
        if (pad) padding = size / 2; 
        
        char *activation_s = option_find_str(options, "activation", "logistic"); 
        ACTIVATION activation = get_activation(activation_s); 
        
        int batch, h, w, c; 
        h = params.h;
        w = params.w; 
        c = params.c; 
        batch = params.batch; 
        if (!(h && w && c)) error("Layer before convolutional layer must output image."); 
        int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0); 
        
        depthwise_convolutional_layer layer = make_depthwise_convolutional_layer(batch, h, w, c, size, stride, padding, activation, batch_normalize); 
        layer.flipped = option_find_int_quiet(options, "flipped", 0); 
        layer.dot = option_find_float_quiet(options, "dot", 0);
        
        return layer;
    }
  
  5) find function 'load_weights_upto' added below into it
    
    if (l.dontload) continue; 
    ++ if (l.type == DEPTHWISE_CONVOLUTIONAL) {
    ++ load_depthwise_convolutional_weights(l, fp);//added by chen
    ++ }

 6) add this function 'load_depthwise_convolutional_weights' into it
   
        //added by chen 
        void load_depthwise_convolutional_weights(layer l, FILE *fp)
        { 
            int num = l.n*l.size*l.size; 
            fread(l.biases, sizeof(float), l.n, fp); 
            if (l.batch_normalize && (!l.dontloadscales)) { 
                fread(l.scales, sizeof(float), l.n, fp); 
                fread(l.rolling_mean, sizeof(float), l.n, fp); 
                fread(l.rolling_variance, sizeof(float), l.n, fp); 
                if (0) { 
                    int i; 
                    for (i = 0; i < l.n; ++i) { 
                        printf("%g, ", l.rolling_mean[i]); 
                    } 
                    printf("\n"); 
                    for (i = 0; i < l.n; ++i) { 
                        printf("%g, ", l.rolling_variance[i]); 
                    } 
                    printf("\n");
                } 
                if (0) { 
                    fill_cpu(l.n, 0, l.rolling_mean, 1); 
                    fill_cpu(l.n, 0, l.rolling_variance, 1);
                }
            }
            fread(l.weights, sizeof(float), num, fp); 
            if (l.flipped) { 
            //transpose_matrix(l.weights, l.c*l.size*l.size, l.n); 
            } 
        #ifdef GPU 
            if (gpu_index >= 0) { 
                push_depthwise_convolutional_layer(l);
            }
        #endif
        }

7) find function 'save_weights_upto' added below into it

       if (l.dontsave) continue;
       ++ if (l.type == DEPTHWISE_CONVOLUTIONAL) {
       ++　　　　save_depthwise_convolutional_weights(l, fp); //added by chen
       ++ }
       
8) add this function 'save_depthwise_convolutional_weights' into it

       //added by chen
       void save_depthwise_convolutional_weights(layer l, FILE *fp)
       { 
       #ifdef GPU 
           if (gpu_index >= 0) { 
               pull_depthwise_convolutional_layer(l);
           } 
       #endif 
           int num = l.n*l.size*l.size;
           fwrite(l.biases, sizeof(float), l.n, fp);
           if (l.batch_normalize) { 
               fwrite(l.scales, sizeof(float), l.n, fp); 
               fwrite(l.rolling_mean, sizeof(float), l.n, fp); 
               fwrite(l.rolling_variance, sizeof(float), l.n, fp);
           } 
           fwrite(l.weights, sizeof(float), num, fp);
       }
Second, open this file /src/network.c .
1) add below into it
    
       #include "data.h"
       ++ #include "depthwise_convolutional_layer.h" // added by chen
       
2) add below into it 

       if(l.type == CONVOLUTIONAL){ 
           resize_convolutional_layer(&l, w, h); 
       } 
       ++ else if(l.type == DEPTHWISE_CONVOLUTIONAL){ 
       ++     resize_depthwise_convolutional_layer(&l, w, h); //added by chen 
       ++ }
       
Third,open this file  /include/darknet.h and add below into it.
   
      BLANK,
    ++ DEPTHWISE_CONVOLUTIONAL //added by chen
    
Final, put these files 'depthwise_convolutional_layer.h depthwise_convolutional_layer.c depthwise_convolutional_kernels.cu' into /src．

# Compile
   Open Makefile add below into it.
   
    ++ OBJ= depthwise_convolutional_layer.o
    ifeq ($(GPU), 1)  
    LDFLAGS+= -lstdc++ 
    ++ OBJ+=depthwise_convolutional_kernels.o 
    endif
     
Make 

    $ make -j8
    
# Test 
     cd darknet
     ./darknet classifier predict cfg/imagenet1k.data cfg/mobilenet_v1.cfg mobilenet_v1_72.weights data/cat.jpg 
     layer     filters    size              input                output
    0 conv     32  3 x 3 / 2   256 x 256 x   3   ->   128 x 128 x  32  0.028 BFLOPs
    1 dw conv     32  3 x 3 / 1   128 x 128 x  32   ->   128 x 128 x  32  0.009 BFLOPs
    2 conv     64  1 x 1 / 1   128 x 128 x  32   ->   128 x 128 x  64  0.067 BFLOPs
    3 dw conv     64  3 x 3 / 2   128 x 128 x  64   ->    64 x  64 x  64  0.005 BFLOPs
    4 conv    128  1 x 1 / 1    64 x  64 x  64   ->    64 x  64 x 128  0.067 BFLOPs
    5 dw conv    128  3 x 3 / 1    64 x  64 x 128   ->    64 x  64 x 128  0.009 BFLOPs
    6 conv    128  1 x 1 / 1    64 x  64 x 128   ->    64 x  64 x 128  0.134 BFLOPs
    7 dw conv    128  3 x 3 / 2    64 x  64 x 128   ->    32 x  32 x 128  0.002 BFLOPs
    8 conv    256  1 x 1 / 1    32 x  32 x 128   ->    32 x  32 x 256  0.067 BFLOPs
    9 dw conv    256  3 x 3 / 1    32 x  32 x 256   ->    32 x  32 x 256  0.005 BFLOPs
    10 conv    256  1 x 1 / 1    32 x  32 x 256   ->    32 x  32 x 256  0.134 BFLOPs
    11 dw conv    256  3 x 3 / 2    32 x  32 x 256   ->    16 x  16 x 256  0.001 BFLOPs
    12 conv    512  1 x 1 / 1    16 x  16 x 256   ->    16 x  16 x 512  0.067 BFLOPs
    13 dw conv    512  3 x 3 / 1    16 x  16 x 512   ->    16 x  16 x 512  0.002 BFLOPs
    14 conv    512  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 512  0.134 BFLOPs
    15 dw conv    512  3 x 3 / 1    16 x  16 x 512   ->    16 x  16 x 512  0.002 BFLOPs
    16 conv    512  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 512  0.134 BFLOPs
    17 dw conv    512  3 x 3 / 1    16 x  16 x 512   ->    16 x  16 x 512  0.002 BFLOPs
    18 conv    512  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 512  0.134 BFLOPs
    19 dw conv    512  3 x 3 / 1    16 x  16 x 512   ->    16 x  16 x 512  0.002 BFLOPs
    20 conv    512  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 512  0.134 BFLOPs
    21 dw conv    512  3 x 3 / 1    16 x  16 x 512   ->    16 x  16 x 512  0.002 BFLOPs
    22 conv    512  1 x 1 / 1    16 x  16 x 512   ->    16 x  16 x 512  0.134 BFLOPs
    23 dw conv    512  3 x 3 / 2    16 x  16 x 512   ->     8 x   8 x 512  0.001 BFLOPs
    24 conv   1024  1 x 1 / 1     8 x   8 x 512   ->     8 x   8 x1024  0.067 BFLOPs
    25 dw conv   1024  3 x 3 / 1     8 x   8 x1024   ->     8 x   8 x1024  0.001 BFLOPs
    26 conv   1024  1 x 1 / 1     8 x   8 x1024   ->     8 x   8 x1024  0.134 BFLOPs
    27 avg                        8 x   8 x1024   ->  1024
    28 conv   1000  1 x 1 / 1     1 x   1 x1024   ->     1 x   1 x1000  0.002 BFLOPs
    29 softmax                                        1000
    Loading weights from ../darknet-mod/mobilenet_v1_72.weights...Done!
    data/cat.jpg: Predicted in 0.005337 seconds.
    43.06%: tiger cat
    17.93%: tabby
    9.49%: Egyptian cat
    3.43%: lynx
    1.38%: bucket

# Valid
    ./darknet classifier valid cfg/imagenet1k.data cfg/mobilenet_v1_416.cfg mobilenet_v1_72.weights 
    
# Precision

network|top1|top5
--------|-----|-------------
Mobilenet_v1|0.7203|0.90514



    
    
