def mixed_operation (exp):  
    exp_list = list(exp)  
    temp = ''  
    behavor_list = []  
    i = 0  
    length = len(exp_list)  
    for item in exp_list:  
        if is_operation(item):  
            behavor_list.append(int(temp))  
            behavor_list.append(item)  
            temp = ''  
        else:  
            temp += item  
  
        if i == length - 1:  
            behavor_list.append(int(temp))  
            break;  
  
        i += 1  
  
    return behavor_list  

def is_operation(oper):  
    if oper == '+' or oper == '-' or oper == '*' or oper == '/':  
        return True  
    else:  
        return False  

def cal(string):
    cin = mixed_operation(string)
    temp=cin[0];
    i=1;
    while(i<len(cin)):
        if cin[i]=='+':
            temp += cin[i+1]
        if cin[i]=='*':
            temp *= cin[i+1]
        i+=2
    return temp
