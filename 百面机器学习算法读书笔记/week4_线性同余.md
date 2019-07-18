伪随机数生成器，线性同余生成器，

生成[0,m)的伪随机数，

```
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
产生伪随机数
"""
def gen(x_i,n,a,c,m):
    x_ip1=(a*x_i+c)%m
    print(x_ip1/m)
    if n>1:
        gen(x_ip1,n-1,a,c,m)

a=1103515245
c=12345
m=2**31-1
        
gen(2,10,5,11,4)
gen(0,10,a,c,m)
```

