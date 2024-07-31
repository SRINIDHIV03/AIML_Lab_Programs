import math

MAX,MIN=1000,-1000

def minimax(depth,nodeIndex,minimizingPlayer,values,alpha,beta,base_case):
    if depth==base_case:
        return values[nodeIndex]
    if minimizingPlayer:
        best=MIN
        for i in range(2):
            val=minimax(depth+1,2*nodeIndex+i,False,values,alpha,beta,base_case)
            best=max(best,val)
            alpha=max(best,alpha)

            if beta<=alpha:
                break
        return best
    else:
        best=MAX
        for i in range(2):
            val=minimax(depth+1,2*nodeIndex+i,True,values,alpha,beta,base_case)
            best=min(best,val)
            beta=min(best,beta)

            if beta<=alpha:
                break
        return best
            
values=[30,1,6,5,1,2,10,20]
base_case=math.ceil(math.log(len(values),2))
optimum=minimax(0,0,True,values,MIN,MAX,base_case)
print("\nOptimum Value: ",optimum)