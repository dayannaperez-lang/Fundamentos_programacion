from src import  multiprocessing_method, threading_method, basic_request_method,asyncio_method
import asyncio
import time

def main():

    sites = [
    "https://picsum.photos/800/600?random=1",
    "https://picsum.photos/800/600?random=2",
    "https://picsum.photos/800/600?random=3",
    "https://picsum.photos/1200/800?random=4",
    "https://picsum.photos/1200/800?random=5",
] * 5
    
    methods = [multiprocessing_method, threading_method, basic_request_method, asyncio_method]
    methods_name = ['multiprocessing_method', 'threading_method', 'basic_request_method', 'asyncio_method']
    methods_duration = []

    for method, name in zip (methods, methods_name): 

                    if name != 'asyncio_method':
                            
                            start_time = time.time()
                            method(sites)   
                            duration = time.time() - start_time 

                            methods_duration.append(duration)    
                            
                            print(f"Downloaded {len(sites)} using {name} in {duration} seconds")

                    else:
                            
                            start_time = time.time()
                            asyncio.run(asyncio_method(sites))  
                            duration = time.time() - start_time    
                            methods_duration.append(duration)   
                            print(f"Downloaded {len(sites)} using {name} in {duration} seconds")   


    evaluation_method = list(zip (methods_duration, methods_name))
    sorted_methods = sorted(evaluation_method, key= lambda x: x[0])   

    print(f""" before evaluating results.... 
          The winner is {sorted_methods[0][1]} with an execution time of {sorted_methods[0][0]}!!. 
          The results are as follows: 
          1.{sorted_methods[0][1]}, {sorted_methods[0][0]}
          2.{sorted_methods[1][1]}, {sorted_methods[1][0]}
          3.{sorted_methods[2][1]}, {sorted_methods[2][0]}
          4.{sorted_methods[3][1]}, {sorted_methods[3][0]} """)


if __name__ == "__main__":
    main()


