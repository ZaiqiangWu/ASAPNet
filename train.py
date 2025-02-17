"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer
import time
import torch

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if (opt.D_steps_per_G == 0):
            trainer.run_generator_one_step(data_i)
        elif (i % opt.D_steps_per_G == 0):
            #start = time.time()
            trainer.run_generator_one_step(data_i)
            #torch.cuda.synchronize(device='cuda')
            #end = time.time()
            #f_time = end - start
            #print("time_%d:%f" % (i, f_time))

        # train discriminator

        if (opt.D_steps_per_G != 0):
            trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses(opt.D_steps_per_G)
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            input_data = data_i[0]
            vm_img = input_data[:,[0,1,2],:,:]
            ssdp_img = input_data[:,[3,4,5],:,:]
            #vm_img
            ground_truth = data_i[1][:,[0,1,2],:,:]
            ground_truth_mask = data_i[1][:, [0, 1, 2], :, :]
            pred_img = trainer.get_latest_generated()[:,[0,1,2],:,:]
            pred_mask = trainer.get_latest_generated()[:,[3],:,:]


            visuals = OrderedDict([('input_vm', vm_img),
                                   ('synthesized_image', pred_img),
                                   ('real_image',ground_truth)])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
