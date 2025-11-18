import numpy as np
import tkinter as tk
from tkinter import ttk,messagebox
from tkinter import *
root=Tk()
root.geometry('500x300')
w='light gray'
root.config(bg=w)
choose=Label(root,bg=w,text='Choose the required gate from the following :-',font=('Comic Sans MS',12))
choose.pack()
gateslist=ttk.Combobox(root,values=['AND','OR','NAND','NOR','XOR','XNOR'])
gateslist.current(0)
gateslist.pack()
input1label=Label(root,bg=w,text='Input Variable 1 :-')
input1label.pack()
input1var=IntVar(value=0)
input10=Radiobutton(root,bg=w,text=0,value=0,variable=input1var)
input10.pack()
input11=Radiobutton(root,bg=w,text=1,value=1,variable=input1var)
input11.pack()
input2label=Label(root,bg=w,text='Input Variable 2 :-')
input2label.pack()
input2var=IntVar(value=0)
input20=Radiobutton(root,bg=w,text=0,value=0,variable=input2var)
input20.pack()
input21=Radiobutton(root,bg=w,text=1,value=1,variable=input2var)
input21.pack()

def calfunc():
    X=np.array([[input1var.get(),input2var.get()]])
    W=np.array([[1,1]])
    net=np.dot(X,W.T)[0][0]
    gate=gateslist.get()
    anbool=0
    result.config(text='',bg=w)
    if gate=='AND':
        if net>=2:
            anbool=1
    elif gate=='OR':
        if net>=1:
            anbool=1
    elif gate=='NAND':
        if net<2:
            anbool=1
    elif gate=='NOR':
        if net<1:
            anbool=1
    elif gate=='XOR':
        if net==1:
            anbool=1
    elif gate=='XNOR':
        if net!=1:
            anbool=1
    else:
        messagebox.showerror('Invalid Gate','Please choose a correct gate')
        return
    result.config(text=f' The Answer of your {gate} gate is {anbool}',bg='black',fg='white')
butt=Button(root,text=' Click To Calculate :-',command=calfunc)
butt.pack()
result=Label(root,text='',font=('Comic Sans MS',20),bg=w)
result.pack()
root.mainloop()

