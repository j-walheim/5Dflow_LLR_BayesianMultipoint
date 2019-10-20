Contains code for Locally Low-Rank Reconstruction and Bayesian multipoint unfolding as described in 

Walheim, Jonas, Hannes Dillinger, and Sebastian Kozerke. "Multipoint 5D flow cardiovascular magnetic resonance-accelerated cardiac-and 
respiratory-motion resolved mapping of mean and turbulent velocities." Journal of Cardiovascular Magnetic Resonance

Locally Low-Rank Reconstruction requires BART. This code assumes linux installation in /usr/local/bin/bart 

Bayesian multipoint unfolding numerically maximizes posterior probability of velocity and intra-voxel standard deviation given the measured data. 
This can be either done using Nelder-Mead Simplex, or using Tensorflow. Both implementations are provided here. They can be executed by running 
either recon.py or recon_tf.py. (As the Simplex based Bayes Unfolding is relatively slow, the data gets reduced to a center slice in systole before
running it, whereas for the TensorFlow Version the whole volume and all frames are reconstructed). 

A dockerfile to set up the environment is provided. However, it was only tested on one workstation so far, if you run into problems using it, please notify me.


Dowload this code to /home/USERNAME/5dllr. We first build the docker image 

	cd /home/USERNAME/5dllr/docker
	docker build -t 5dllr/5dllr:latest .


Of note, depending on your cuda installation you may need to adapt the environment variable ENV LD_LIBRARY_PATH in docker/dockerfile before building the image to 
make TensorFlow work. (on my machine it was /usr/local/cuda/extras/CUPTI/lib64)

	ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

If the data for the demo code have not been downloaded to the program directory, we do so now

    cd /home/USERNAME/5dllr/
    wget --header 'Host: files.de-1.osf.io' 'https://files.de-1.osf.io/v1/resources/36gdr/providers/osfstorage/5d5bf05c0e88ae0018c0981f?action=download&version=1&direct' --output-document 'data.h5'


Then we start the image, mapping the home directory from the image system to the home folder in the image
		
	docker run -d --runtime=nvidia --volume="/home/USERNAME:/home/USERNAME" 5dllr/5dllr

Get <image_name> using 

	docker ps

and run recon in the image

	docker exec -it <image_name> python3 /home/USERNAME/5dllr/recon.py
