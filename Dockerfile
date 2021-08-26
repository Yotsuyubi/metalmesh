FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y wget vim git
RUN apt-get install -y python3 

ADD build_meep_python.sh /home
RUN /bin/bash /home/build_meep_python.sh

RUN echo "LS_COLORS=\$LS_COLORS:'di=0;93:' ; export LS_COLORS" >> ~/.bashrc
RUN echo 'export export LD_LIBRARY_PATH="/usr/local/lib:/usr/lib/x86_64-linux-gnu/hdf5/serial" \
    export LDFLAGS="-L/usr/local/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial" \
    export CPPFLAGS="-I/usr/local/include -I/usr/include/hdf5/serial" \
    export PYTHONPATH="$HOME/install/meep/python" \
    export GUILE_WARN_DEPRECATED="no" \
    export MPLBACKEND=Agg' >> ~/.bashrc

RUN pip3 install scikit-image pillow torch torchvision tqdm

RUN mkdir /home/meep
VOLUME /home/meep

WORKDIR /home/meep