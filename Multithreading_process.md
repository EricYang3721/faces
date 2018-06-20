# Multithreading

For a given list of numbers print square and cube of every number. 
For example: 
Input: [2,3,8,9]
Output: square list - [4,9,64,81], cube list-[8,27,512,729]
##### Code

```python
import time
import threading

def calc_square(numbers):
    print("calculate square numbers")
    for n in numbers:
        time.sleep(0.2)
        print("square:", n*n)
      
def calc_cube(numbers):
    print('calculate cube numbers')
    for n in numbers:
        time.sleep(0.2)
        print('cube', nnn)

arr = [2,3,8,9]
t = time.time()
t1=threading.Thread(target=calc_square, args=(arr,))  # , is necessary, since it takes an tuple as input
t2=threading.Thread(target=calc_cube, args=(arr,))
'''Main function is the main thread, and t1 and t2 are 2 threads. start() allows it to run 2 threads, and join() make the threads go back to main thread'''
t1.start() # start
t2.start() # excute 2 in parallel
t1.join() #wait until t1 is done
t2.join() 
calc_square(arr)# non multithreading code
calc_cube(arr)# non multithreading code
print('done', time.time()-t)
```



# Multiprocessing


##### Code
```python
import time
import multiprocessing

def calc_square(numbers):
    for n in numbers:
        print('square ' + str(n*n))    # not necessary to convert int into string

def calc_cube(numbers):
    for n in numbers:
        print('cube ' + str(nnn))

if __name__ == "__main__":
    arr = [2,3,8]
    p1 = multiprocessing.Process(target=calc_square, args=(arr,))  # , is necessary, tuple is input
    p2 = multiprocessing.Process(target=calc_cube, args=(arr,))
    p1.start()
	p2.start()
	p1.join() 
	p2.join()
	print("Done!")

```



##### Problem example  (global variable)

```python
import time
import multiprocessing

square_result=[]
def calc_square(numbers):
    global square_results
    for n in numbers:        
        square_result.append(n*n)

if name == "main":
    arr = [2,3,8]
    p1 = multiprocessing.Process(target=calc_square, args=(arr,))
    p1.start()
    p1.join()

print('results' + str(square_result))
print("Done!")
```
This code doesn't print anything.

Every process has its own address space (virtual memory). Thus program variables are not shared between two processes. You need to use interprocess communication (IPC) techniques if you want to share data between two processes. In this example, global variable square_results only exists in p1 thread, it doesn't return back to main thread. If print within the process, it will work.

Main difference between multiprocessing and multithreading. It will print in multithreading but not multiprocessing. Processes doesn't share data, but different threading does.



# Sharing Data Between Processes Using Array and Variable

##### Code

```python
import multiprocessing

result=[]
def calc_square(numbers):
    global result
    for n in numbers:
        print('square ', n*n)
        result.append(n*n)
    print('inside process' + str(result))


if __name__ == "__main__":
    arr = [2,3,8]
    p = multiprocessing.Process(target=calc_square, args=(arr,))
    p.start()
    p.join()
    
    print('outside process ' + str(result))
```

##### result: 

```
square  4
square  9
square  64
inside process[4, 9, 64]
outside process []
```

Reasons are described in the section above.

### Method to solve it  --> use shared memory

2 ways to share: **value** and **array**

##### Code

```python
import multiprocessing

def calc_square(numbers, result, v):
    v.value = 5.67  # note v.value, not v
    for idx, n in enumerate(numbers): 
        # shared memory support doesn't suppot append
        # get index and value both
        result[idx] = n*n

if __name__ == "__main__":
    arr = [2,3,8]
    
    # Shared memory, array and value
    result = multiprocessing.Array('i', 3) # integer, size 3
    v = multiprocessing.Value('d', 0.0)
    p = multiprocessing.Process(target=calc_square, args=(arr, result, v))
    p.start()
    p.join()
    
    # different method to print shared array
    print(result[:])
    print(v.value)
```

##### result

```
[4, 9, 64]
5.67
```



# Share Data Between Processes Using Queue

Queue is basically a shared memory

Multiprocessing Queue is different from Queue modele in Python. 

Multiprocessing Queue: lives in shared memory, used to share data between processes

Queue Module: lives in in-process memory, used to share data between threads

##### Code 

```python
import multiprocessing

def calc_square(numbers, q):
    for n in numbers:
        q.put(n*n)

if __name__ == "__main__":
    arr = [2,3,8]
    q = multiprocessing.Queue() # Queue data structure, first in first out
    # functions queue.put(), queue.get()
    
    p = multiprocessing.Process(target=calc_square, args=(arr, q))
    p.start()
    p.join()
    
    while q.empty() is False:
        print(q.get())    
```



# Multiprocessing Lock

##### code (problematic)

```python
import time
import multiprocessing

def deposit(balance, lock):
    for i in range(100):
        time.sleep(0.01)
        balance.value = balance.value + 1

def withdraw(balance, lock):
    for i in range(100):
        time.sleep(0.01)
        balance.value = balance.value - 1

if __name__ == '__main__':
    balance = multiprocessing.Value('i', 200)
    # not using lock
    d = multiprocessing.Process(target=deposit, args=(balance,))
    w = multiprocessing.Process(target=withdraw, args=(balance,))
    d.start()
    w.start()
    d.join()
    w.join()
    print(balance.value)
```

Output is different every time, due to multiprocessing on the shared variable balance. Hence, lock needs to be used. 

```python
import time
import multiprocessing

def deposit(balance, lock):
    for i in range(100):
        time.sleep(0.01)
        lock.acquire() # lock acquire
        balance.value = balance.value + 1
        lock.release() # release lock

def withdraw(balance, lock):
    for i in range(100):
        time.sleep(0.01)
        lock.acquire()  # acquire lock
        balance.value = balance.value - 1
        lock.release()  # release lock

if __name__ == '__main__':
    balance = multiprocessing.Value('i', 200)
    lock = multiprocessing.Lock() # Define a lock
    d = multiprocessing.Process(target=deposit, args=(balance,lock)) # pass lock in process
    w = multiprocessing.Process(target=withdraw, args=(balance,lock))
    d.start()
    w.start()
    d.join()
    w.join()
    print(balance.value)
```



# Multiprocessing pool

##### Code

```python
from multiprocessing import Pool

def f(n):
    return n*n

if __name__ == "__main__":
    p = Pool(processes=3)  #
    result = p.map(f,[1,2,3,4,5])  # divide work among cores of cpu
    for n in result:
        print(n)
```

```python
from multiprocessing import Pool
import time

def f(n):
    sum = 0
    for x in range(1000):
		sum += x*x
    return sum

if __name__ == "__main__":
    t1 = time.time()
    p = Pool()  # processes = 3 mean use 3 processers
    result = p.map(f,range(10000))  # divide work among cores of cpu
   	p.close()
    p.join()
    print("Pool took: ", time.time()-t1)
    
    t2=time.time()
    result = []
    for x in range(10000):
        result.append(f(x))
    print("Serial processing took: ", time.time()-t2)
```

##### Results:

```python
Pool took:  0.33030104637145996
Serial processing took:  0.8223261833190918
```

