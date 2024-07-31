import numpy as np

def hill_climbing(func,start,step_size=0.01,max_iters=1000):
    current_state=start
    current_value=func(current_state)

    

    for _ in range(max_iters):
        next_positive_postion=current_state+step_size
        next_positive_value=func(next_positive_postion)

        next_negative_position=current_state-step_size
        next_negative_value=func(next_negative_position)

        if next_positive_value>current_value and next_positive_value>=next_negative_value:
            current_state=next_positive_postion
            current_value=next_positive_value
        elif next_negative_value>current_value and next_negative_value>next_positive_value:
            current_state=next_negative_position
            current_value=next_negative_value
        else:
            break
    return current_state,current_value

while True:
    try:
        func_str=input("Enter function of x: ")
        x=0
        eval(func_str)
        break
    except Exception as e:
        print("\nInvalid Function. Try Again")
func=lambda x:eval(func_str)
while True:
    try:
        start=input("Enter start position: ")
        start=float(start)
        break
    except Exception as e:
        print("Invalid input. Try Again")
maxima,max_value=hill_climbing(func,start)
print("Maxima occurs at: ",maxima)
print("Maximum Value: ",max_value)





