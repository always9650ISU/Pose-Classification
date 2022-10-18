from multiprocessing import Process, Queue, Value
from time import sleep


def a1 (q, v):

    while True:
        sleep(0.1)
        print(v.value)
        if v.value == 10:
            break

def a2(q, v):

    a = 0
    while a < 100:
        sleep(0.01)
        a += 1
        if a > 10:
            v.value =  a // 10
def a3(q, v=0, a=1):
    while True:
        if not q:
            print("***")

        if v.value ==10:
            print("Close", a)
            break

if __name__ == "__main__":
    
    
    q = Queue()
    v = Value('i', 1)
    d = {"v":v,"a":10, 'c':3}
    p1 = Process(target=a1, args=(q, v))
    p2 = Process(target=a2, args=(q, v))
    p3 = Process(target=a3, args=(q, ), kwargs={"v":v,"a":10})
    p1.start()
    p2.start()
    p3.start()