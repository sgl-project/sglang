switching from regular model serving to our model serving would require minimal effort
for example by passing in 
--heter-precision-config heter_config.json/yaml

the heter_config should include the configs from model.md, also including the locations of model weights

the entire implementation should be naturally supporting torch.compile as well as cudagraph
