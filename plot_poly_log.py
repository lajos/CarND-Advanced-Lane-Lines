import constants, utils
import numpy as np
import matplotlib.pyplot as pyplot

poly_log = np.float32(np.array(utils.read_csv(constants.poly_log_name)))

#pyplot.plot(poly_log[:,0])
#pyplot.plot(poly_log[:,1])
#pyplot.plot(poly_log[:,2])


#pyplot.plot(poly_log[:,3])
#pyplot.plot(poly_log[:,4])
#pyplot.plot(poly_log[:,5])

pyplot.plot(np.subtract(poly_log[:,0], np.roll(poly_log[:,0],1)))
#pyplot.plot(np.subtract(poly_log[:,1], np.roll(poly_log[:,1],1)))
#pyplot.plot(np.subtract(poly_log[:,2], np.roll(poly_log[:,2],1)))

#pyplot.plot(np.subtract(poly_log[:,3], np.roll(poly_log[:,3],1)))
# pyplot.plot(np.subtract(poly_log[:,4], np.roll(poly_log[:,4],1)))
# pyplot.plot(np.subtract(poly_log[:,5], np.roll(poly_log[:,5],1)))

pyplot.show()


