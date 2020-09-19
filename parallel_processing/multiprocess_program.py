from multiprocessing import Pool
import time
def  f(x):
    time.sleep(10)
    return x*x


if __name__ == '__main__':
    pool = Pool(processes=10)              # start 4 worker processes
    result = pool.apply_async(f, [10])    # evaluate "f(10)" asynchronously
    print (result.get())           # prints "100" unless your computer is *very* slow
    print (pool.map(f, range(10)))         # prints "[0, 1, 4,..., 81]"