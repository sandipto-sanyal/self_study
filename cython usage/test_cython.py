import time
import cython_test

number = 100000

start = time.time()
cython_test.test(number)
end =  time.time()

cy_time = end - start
print("Cython time = {}".format(cy_time))