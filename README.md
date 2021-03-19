# Multi-Step-Deformable-Registration

This is a PyTorch implementation of the following paper:
'Unsupervised Multi-Step Deformable Registration of Remote Sensing Imagery based on Deep Learning', Remote Sensing

Please cite the this publication if you use this code.

In this repository we have applied the multi-step registration network on the MNIST dataset. Since the MNIST images are of lower dimensions we employ a more simple architecture than the one used in the paper. The multi-step process however remains the same.

   ```ruby 
    def forward(self, moving, reference):

        deformed_images = []

        input = torch.cat( (moving, reference), 1)

        for s in range(0,self.step):
            if s==0:
                e1,e2,e3 = self.encoder(input)
                d = self.decoder(e1,e2,e3)
                deformable = self.deform(d)
                out_info = smoothTransformer2D([moving, deformable], self.c)
                deformed, sgrid = out_info
                deformed_images.append(deformed)
            else:
                new_input = torch.cat((previous_deformed,reference), 1)
                e1,e2,e3 = self.encoder(new_input)
                d = self.decoder(e1,e2,e3)
                deformable = deformable + self.deform(d)
                out_info = smoothTransformer2D([moving, deformable, previous_sgrid], self.c)
                deformed, sgrid = out_info
                deformed_images.append(deformed)

            previous_deformed = deformed
            previous_sgrid = sgrid

        return deformable, sgrid, deformed_images

   ```

# To use this code:
Run 'python train.py' to train the model.  <br/> <br/>
Run 'show.py' to make registration predictions on the test set.

# Version of employed packages:
- CUDA (10.2)
- pytorch 1.5.1
- python 3.6.10
- matplotlib 3.2.2

# Outputs:
Examples of the output of the show.py script (result after few training epochs):

![example1](/output/__1to5.png) 


![example1](/output/__4to7.png)

![example1](/output/__1to0.png)
