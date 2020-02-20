# @Date:   2019-12-31T13:03:11+01:00
# @Last modified time: 2019-12-31T13:23:43+01:00

import numpy as np
from keras.callbacks import LearningRateScheduler

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=2):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        print(initial_lr * (decay_factor ** np.floor(epoch/step_size)))
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    return LearningRateScheduler(schedule)

# lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)
#https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
# model.fit(X_train, Y_train, callbacks=[lr_sched])
