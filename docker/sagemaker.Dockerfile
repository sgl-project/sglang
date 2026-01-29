FROM lmsysorg/sglang:latest

COPY serve /usr/bin/serve
RUN chmod 777 /usr/bin/serve

# Install missing dependencies for multimodal backend
RUN python3 -m pip install "imageio==2.36.0" "diffusers==0.36.0" remote-pdb

ENTRYPOINT [ "/usr/bin/serve" ]
