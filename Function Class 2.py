# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sympify, lambdify
from sympy.abc import x

class Function(object):
    
    def __init__(self):
        self.functions = {}
        self.variables = {}
        self.step = {}
        self.fig = plt.figure()
    
    def addVariable(self, name):
        self.variables[name] = []
    
    def addRange(self, variable, minR, maxR, step):
        self.step[variable] = step
        rang = np.arange(minR,maxR+step,step)
        for i in range(len(rang)):
            rang[i] = round(rang[i], int(np.ceil(abs(np.log10(step)))))
        self.variables[variable] = rang
    
    def retrieveRange(self, variable):
        try:
            return self.variables[variable]
        except:
            raise ValueError('Not in Dictionary')
    
    def removeRange(self,variable):
        try:
            del self.variables[variable]
        except:
            raise ValueError('Not in Dictionary')
    
    def lamdifyFunction(self, function):
        myfunction=sympify(function)
        mylambdifiedfunction=lambdify(x,myfunction,'numpy')
        return mylambdifiedfunction
    
    def addFunction(self, function, variable):
        mylambdifiedfunction = self.lamdifyFunction(function)
        self.functions[function] = variable, self.variables[variable], mylambdifiedfunction(self.variables[variable])
    
    def removeFunction(self, function):
        del self.functions[function]
        
    def strFunction(self, function):
        string = str(function)
        i = 0
        while i < len(string):
            if string[i] == '*' and string[i+1] == '*':
                string = string[:i] + '^' + string[i+2:]
                i -= 2
            i += 1
        i = 0
        while i < len(string):
            if string[i] == '*':
                string = string[:i] + string[i+1:]
                i -= 1
            i += 1
        return string
    
    def asymFunction(self, function, ylist):
        step = self.step[self.functions[function][0]]
        step1 = round(step, int(np.ceil(abs(np.log10(step)))))
        maxY, minY = int(1/(step1)), int(-1/(step1))
        ylist[ylist>maxY] = np.inf
        ylist[ylist<minY] = -np.inf
        return ylist
        
    def Graph(self, function = 'NaN', minInd = 0, maxInd = 0):
        if function != 'NaN':
            x,y = list(self.functions[function])[1], list(self.functions[function])[2]
            y = self.asymFunction(function, y)
            plt.plot(x,y, label = self.strFunction(function))
            plt.legend(bbox_to_anchor =(0.75, 1.3), ncol = 2, fontsize = 8)
        if minInd != 0 or maxInd != 0:
            functions = list(self.functions.keys())
            for i in range(minInd, maxInd+1):
                x,y = list(self.functions.values())[i][1], list(self.functions.values())[i][2]
                y = self.asymFunction(function, y)
                plt.plot(x,y, label = self.strFunction(functions[i]))
        plt.xlabel('x-axis')
        plt.ylabel('y-axis')
        plt.title('Graphs of f(x)')
        plt.axhline(y=0, c='k', alpha = 0.5)
        plt.axvline(x=0, c='k', alpha = 0.5)
        plt.legend(bbox_to_anchor =(0.75, 1.3), ncol = 2, fontsize = 8)
        
    def approxZero(self, function, graph = 'off'):
        x,y = list(self.functions[function])[1], list(self.functions[function])[2]
        y = self.asymFunction(function, y)
        x1, y1 = [], []
        step = self.step[self.functions[function][0]]
        for i in range(len(x)):
            if abs(y[i]) < step:
                x1.append(x[i])
                y1.append(y[i])
        if graph == 'on':
            self.Graph(function)
            plt.plot(x1, y1, 'ko', label = 'Zeros of ' + self.strFunction(function))
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.axhline(y=0, c='k', alpha = 0.5)
            plt.axvline(x=0, c='k', alpha = 0.5)
            plt.legend(bbox_to_anchor =(0.75, 1.3), ncol = 2, fontsize = 8)
        return 'Zeros of ' + self.strFunction(function) + ' are at ' + str(x1)
            
    
    def approxInter(self, function1, function2, graph = 'off'):
        if (function1 in self.functions) and (function2 in self.functions):
            x,y = list(self.functions[function1])[1][:],list(self.functions[function2])[1][:]
            x1,y1 = list(self.functions[function1])[2][:],list(self.functions[function2])[2][:]
            step1 = self.step[self.functions[function1][0]]
            step2 = self.step[self.functions[function2][0]]
            step = (step1+step2)/2
            values = []
            values1 = []
            values2 = []
            for i in range(len(x1)):
                for j in range(len(y1)):
                    if abs(x1[i]-y1[j]) < step:
                        values.append([i,j,x1[i],y1[j]])
            for l in range(len(values)):
                if abs(x[values[l][0]] - y[values[l][1]]) < step:
                    values1.append((x[values[l][0]] + y[values[l][1]])/2)
                    values2.append((values[l][2] + values[l][3])/2)
            if graph == 'on':
                self.Graph(function1)
                self.Graph(function2)
                plt.plot(values1, values2, 'ko', label = 'Intercepts')
                plt.xlabel('x-axis')
                plt.ylabel('y-axis')
                plt.axhline(y=0, c='k', alpha = 0.5)
                plt.axvline(x=0, c='k', alpha = 0.5)
                plt.legend(bbox_to_anchor =(0.75, 1.3), ncol = 2, fontsize = 8)
            return 'Points of intersection of ' + self.strFunction(function1) + ' and ' + self.strFunction(function2) + ' are ' + str(values1)
        else:
            raise ValueError('Functions not in Dictionary')
    
    def approxExt(self, function, graph = 'off'):
        x = list(self.functions[function])[1][:]
        y = list(self.functions[function])[2][:]
        step = self.step[self.functions[function][0]]
        y1 = sorted(y)
        extrema = []
        x2 = []
        y2 = []
        for i in range(len(y)):
            if abs(y1[0] - y[i]) < step/100:
                extrema.append('Min at '+ str(x[i]))
                x2.append(x[i])
                y2.append(y[i])
            if abs(y1[len(y1)-1] - y[i]) < step/100:
                extrema.append('Max at ' + str(x[i]))
                x2.append(x[i])
                y2.append(y[i])
        if graph == 'on':
            self.Graph(function)
            plt.plot(x2,y2,'ko', label = 'Extremas of ' + self.strFunction(function))
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.axhline(y=0, c='k', alpha = 0.5)
            plt.axvline(x=0, c='k', alpha = 0.5)
            plt.legend(bbox_to_anchor =(0.75, 1.3), ncol = 2, fontsize = 8)
        return extrema
    
    def approxDer(self, function, x, graph = 'off'):
        x1 = list(self.functions[function])[1][:]
        step = self.step[self.functions[function][0]]
        mylambdifiedfunction = self.lamdifyFunction(function)
        der = ((mylambdifiedfunction(x+step) - mylambdifiedfunction(x))/(step))
        if graph == 'on':
            self.Graph(function)
            plt.plot(x1, der*(x1-x)+mylambdifiedfunction(x), label = 'Approximate Tangent Line of ' + self.strFunction(function) + ' at x = ' + str(x))
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.axhline(y=0, c='k', alpha = 0.5)
            plt.axvline(x=0, c='k', alpha = 0.5)
            plt.legend(bbox_to_anchor =(0.75, 1.3), ncol = 2, fontsize = 8)
        return der
    
    def approxDerNum(self, function, x, graph = 'off'):
        '''https://math.mit.edu/~djk/calculus_beginners/chapter09/section01.html'''
        x1 = list(self.functions[function])[1][:]
        step = self.step[self.functions[function][0]]
        mylambdifiedfunction = self.lamdifyFunction(function)
        def C(function, a, x, step):
            der = (mylambdifiedfunction(x+a*step) - mylambdifiedfunction(x-a*step))/(2*a*step)
            return der
        def D(function, a, x, step):
            der = ((4*C(mylambdifiedfunction, a, x, step)) - C(mylambdifiedfunction, 2*a, x, step))/3
            return der
        def E(function, a, x, step):
            der = (16*D(mylambdifiedfunction, a, x, step) - D(mylambdifiedfunction, 2*a, x, step))/15
            return der
        if graph == 'on':
            self.Graph(function)
            plt.plot(x1, E(function, 1, x, step)*(x1-x)+mylambdifiedfunction(x), label = 'Approximate Tangent Line of ' + self.strFunction(function) + ' at ' + str(x))
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.axhline(y=0, c='k', alpha = 0.5)
            plt.axvline(x=0, c='k', alpha = 0.5)
            plt.legend(bbox_to_anchor =(0.75, 1.3), ncol = 2, fontsize = 8)
        return E(function, 1, x, step)
    
    def fDerGraph(self, function, minX = 'NaN', maxX = 'NaN', graph = 'on'):
        x = list(self.functions[function])[1][:]
        x1, y1 = [], []
        if minX == 'NaN' and maxX == 'NaN':
            minX = x[0]
            maxX = x[len(x)-1]
            x1 = x
        else:
            for i in range(len(x)):
                if x[i] <= maxX and x[i] >= minX:
                    x1.append(x[i])
        for i in range(len(x1)):
            y1.append(self.approxDerNum(function, x1[i]))
        y1 = np.array(y1)
        if graph != 'off':
            y1 = self.asymFunction(function, y1)
            plt.plot(x1, y1, label = 'First Derivative Function of ' + self.strFunction(function))
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.axhline(y=0, c='k', alpha = 0.5)
            plt.axvline(x=0, c='k', alpha = 0.5)
            plt.legend(bbox_to_anchor =(0.75, 1.3), ncol = 2, fontsize = 8)
        return x1, y1
    
    def approxSDer(self, function, x, graph = 'off'):
        x1,y1 = self.fDerGraph(function, graph = 'off')
        y2, y3 = [], []
        step = self.step[self.functions[function][0]]
        for i in range(len(x1)):
            if abs(x1[i] - x) < step:
                y2.append((y1[i+1]-y1[i-1])/(x1[i+1]-x1[i-1]))
                y3.append(y1[i])
        der, yval = sum(y2)/len(y2), sum(y3)/len(y3)
        if graph == 'on':
            self.fDerGraph(function)
            plt.plot(x1, der*(x1 - x) + yval, label = 'Approximate Second Derivative Tangent Line of ' + self.strFunction(function) + ' at ' + str(x))
            plt.axhline(y=0, c='k', alpha = 0.5)
            plt.axvline(x=0, c='k', alpha = 0.5)
            plt.legend(bbox_to_anchor =(0.75, 1.3), ncol = 2, fontsize = 6)
        return der
            
    
    def sDerGraph(self, function, minX = 'NaN', maxX = 'NaN', graph = 'on'):
        x = list(self.functions[function])[1][:]
        x1, y1, y2 = [], [], []
        if minX == 'NaN' and maxX == 'NaN':
            minX = x[0]
            maxX = x[len(x)-1]
            x1 = x
        x1, y1 = self.fDerGraph(function, minX, maxX, 'off')
        for i in range(1,len(x1)-1):
            y2.append((y1[i+1]-y1[i-1])/(x1[i+1]-x1[i-1]))
        y2 = np.array(y2)
        x2 = x1[1:len(x1)-1]
        if graph != 'off':
            y2 = self.asymFunction(function, y2)
            plt.plot(x2, y2, label = 'Second Derivative Function of ' + self.strFunction(function))
            plt.xlabel('x-axis')
            plt.ylabel('y-axis')
            plt.axhline(y=0, c='k', alpha = 0.5)
            plt.axvline(x=0, c='k', alpha = 0.5)
            plt.legend(bbox_to_anchor =(0.75, 1.3), ncol = 2, fontsize = 8)
        return x2, y2
    
    def fDerTest(self, function, minX = 'NaN', maxX = 'NaN'):
        x = list(self.functions[function])[1][:]
        y = list(self.functions[function])[2][:]
        step = self.step[self.functions[function][0]]
        x1, y1 = [], []
        if minX == 'NaN' and maxX == 'NaN':
            minX = x[0]
            maxX = x[len(x)-1]
            x1 = x
            y1 = y
        else:
            for i in range(len(x)):
                if x[i] <= maxX and x[i] >= minX:
                    x1.append(x[i])
                    y1.append(y[i])
        Ext, Inc, Dec = [],[],[]
        x2, y2 = self.fDerGraph(function, minX, maxX, 'off')
        for i in range(len(x2)):
            if abs(y2[i]) < step:
                Ext.append(i)
        if len(Ext) == 0:
            if y2[0] < 0 and y2[len(y2)-1] < 0:
                Dec.append([minX, maxX])
            elif y2[0] > 0 and y2[len(y2)-1] > 0:
                Inc.append([minX, maxX]-1)
        elif len(Ext) == 1:
            print(y2[Ext[0]+1], y2[len(y2)-1])
            if y2[0] < 0 and y2[Ext[0]-1] < 0:
                Dec.append([x1[0], x1[Ext[0]-1]])
            elif y2[0] > 0 and y2[Ext[0]-1] > 0:
                Inc.append([x1[0], x1[Ext[i]-1]])
            if y2[Ext[0]+1] < 0 and y2[len(y2)-1] < 0:
                Dec.append([x1[Ext[0]-1], x1[len(x1)-1]])
            elif y2[Ext[0]+1] > 0 and y2[len(y2)-1] > 0:
                Inc.append([x1[Ext[0]-1], x1[len(x1)-1]])
        else:
            for i in range(len(Ext)):
                if i == 0:
                    if y2[0] < 0 and y2[Ext[i]-1] < 0:
                        Dec.append([x1[0],x1[Ext[i]]])
                    elif y2[0] > 0 and y2[Ext[i]-1] > 0:
                        Inc.append([x1[0],x1[Ext[i]]])
                elif i == (len(Ext)-1):
                    if y2[Ext[i]] < 0 and y2[len(y2)-1] < 0:
                        Dec.append([x1[Ext[i]],x1[len(y2)-1]])
                    elif y2[Ext[i]] > 0 and y2[len(y2)-1] > 0:
                        Inc.append([x1[Ext[i]],x1[len(y2)-1]])
                else:
                    if y2[Ext[i]+1] < 0 and y2[Ext[i+1]-1] < 0:
                        Dec.append([x1[Ext[i]],x1[Ext[i+1]]])
                    elif y2[Ext[i]+1] > 0 and y2[Ext[i+1]-1] > 0:
                        Inc.append([x1[Ext[i]],x1[Ext[i+1]]])
        Ext1, Inc1, Dec1 = 'Extrema At','f(x) is increasing in the interval(s)','f(x) is decreasing in the interval(s)'
        for item in Ext:
            Ext1 += ' ' + str([x1[item],y1[item]])
        for item in Inc:
            Inc1 += ' ' + str(item)
        for item in Dec:
            Dec1 += ' ' + str(item)
        return Ext1, Inc1, Dec1
                
    def sDerTest(self, function, minX = 'NaN', maxX = 'NaN'):
        x = list(self.functions[function])[1][:]
        y = list(self.functions[function])[2][:]
        step = self.step[self.functions[function][0]]
        x1, y1 = [], []
        if minX == 'NaN' and maxX == 'NaN':
            minX = x[0]
            maxX = x[len(x)-1]
            x1 = x
            y1 = y
        else:
            for i in range(len(x)):
                if x[i] <= maxX and x[i] >= minX:
                    x1.append(x[i])
                    y1.append(y[i])
        Inf, CCU, CCD = [],[],[]
        x2, y2 = self.sDerGraph(function, minX, maxX, 'off')
        for i in range(len(x2)):
            if abs(y2[i]) < step:
                Inf.append(i)
        if len(Inf) == 0:
            if y2[0] < 0 and y2[len(y2)-1] < 0:
                CCD.append([minX, maxX])
            elif y2[0] > 0 and y2[len(y2)-1] > 0:
                CCU.append([minX, maxX])
        elif len(Inf) == 1:
            print(y2[Inf[0]+1], y2[len(y2)-1])
            if y2[0] < 0 and y2[Inf[0]-1] < 0:
                CCD.append([x1[0], x1[Inf[0]-1]])
            elif y2[0] > 0 and y2[Inf[0]-1] > 0:
                CCU.append([x1[0], x1[Inf[0]-1]])
            if y2[Inf[0]+1] < 0 and y2[len(y2)-1] < 0:
                CCD.append([x1[Inf[0]-1], x1[len(x1)-1]])
            elif y2[Inf[0]+1] > 0 and y2[len(y2)-1] > 0:
                CCU.append([x1[Inf[0]-1], x1[len(x1)-1]])
        else:
            for i in range(len(Inf)):
                if i == 0:
                    if y2[0] < 0 and y2[Inf[i]-1] < 0:
                        CCD.append([x1[0],x1[Inf[i]]])
                    elif y2[0] > 0 and y2[Inf[i]-1] > 0:
                        CCU.append([x1[0],x1[Inf[i]]])
                elif i == (len(Inf)-1):
                    if y2[Inf[i]] < 0 and y2[len(y2)-1] < 0:
                        CCD.append([x1[Inf[i]],x1[len(y2)-1]])
                    elif y2[Inf[i]] > 0 and y2[len(y2)-1] > 0:
                        CCU.append([x1[Inf[i]],x1[len(y2)-1]])
                else:
                    if y2[Inf[i]+1] < 0 and y2[Inf[i+1]-1] < 0:
                        CCD.append([x1[Inf[i]],x1[Inf[i+1]]])
                    elif y2[Inf[i]+1] > 0 and y2[Inf[i+1]-1] > 0:
                        CCU.append([x1[Inf[i]],x1[Inf[i+1]]])
        Inf1, CCU1, CCD1 = 'Point of Inflection at', 'f(x) is concave up in the interval(s)', 'f(x) is concave down in the interval(s)'
        for item in Inf:
            Inf1 += ' ' + str([x1[item], y1[item]])
        for item in CCU:
            CCU1 += ' ' + str(item)
        for item in CCD:
            CCD1 += ' ' + str(item)
        return Inf1, CCU1, CCD1
        
    def leftsum(self, function, a, b, graph = 'off'):
        x = list(self.functions[function])[1][:]
        y = list(self.functions[function])[2][:]
        step = self.step[self.functions[function][0]]
        x1, y1 = [], []
        approx = 0
        if x[0] > a or x[len(x)-1] < b:
            return('Enter bounds not in the range of the variable')
        for i in range(len(x)):
            if x[i] <= b and x[i] >= a:
                x1.append(x[i])
                y1.append(y[i])
        for j in range(len(x1)-1):
            approx += step * y1[j]
        if graph == 'on':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for k in range(len(x1)-1):
                x2 = [x1[k],x1[k],x1[k+1],x1[k+1]]
                y2 = [0,y1[k],y1[k],0]
                ax.fill(x2,y2,'c',edgecolor='c',alpha=0.5)
            plt.axhline(y=0, c='k', alpha = 0.5)
            plt.axvline(x=0, c='k', alpha = 0.5)
            self.Graph(function)
        return 'Approximate left riemann sum of ' + self.strFunction(function) + ' from ' + str(a) + ' to ' + str(b) + ' is ' + str(approx)
    
    def rightsum(self, function, a, b, graph = 'off'):
        x = list(self.functions[function])[1][:]
        y = list(self.functions[function])[2][:]
        step = self.step[self.functions[function][0]]
        x1, y1 = [], []
        approx = 0
        for i in range(len(x)):
            if x[i] <= b+step and x[i] >= a-step:
                x1.append(x[i])
                y1.append(y[i])
        for j in range(1,len(x1)):
            approx += step * y1[j]
        if graph == 'on':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for k in range(len(x1)-1):
                x2 = [x1[k],x1[k],x1[k+1],x1[k+1]]
                y2 = [0,y1[k+1],y1[k+1],0]
                ax.fill(x2,y2,'c',edgecolor='c',alpha=0.5)
            plt.axhline(y=0, c='k', alpha = 0.5)
            plt.axvline(x=0, c='k', alpha = 0.5)
            self.Graph(function)
        return 'Approximate right riemann sum of ' + self.strFunction(function) + ' from ' + str(a) + ' to ' + str(b) + ' is ' + str(approx)
    
    def trapsum(self, function, a, b, graph = 'off'):
        x = list(self.functions[function])[1][:]
        y = list(self.functions[function])[2][:]
        step = self.step[self.functions[function][0]]
        x1, y1 = [], []
        approx = 0
        for i in range(len(x)):
            if x[i] <= b+step and x[i] >= a-step:
                x1.append(x[i])
                y1.append(y[i])
        for j in range(1,len(x1)-1):
            approx += step * (y1[j] + y1[j+1])/2
        if graph == 'on':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for k in range(len(x1)-1):
                x2 = [x1[k],x1[k],x1[k+1],x1[k+1]]
                y2 = [0,y1[k],y1[k+1],0]
                ax.fill(x2,y2,'c',edgecolor='c',alpha=0.5)
            plt.axhline(y=0, c='k', alpha = 0.5)
            plt.axvline(x=0, c='k', alpha = 0.5)
            self.Graph(function)
        return 'Approximate trapezoidal riemann sum of ' + self.strFunction(function) + ' from ' + str(a) + ' to ' + str(b) + ' is ' + str(approx)
    
    def __str__(self):
        return str(self.functions)



c = Function()
c.addVariable('x')
c.addVariable('a')
c.addRange('x',-5.0, 5.0,0.1)
c.addFunction(x, 'x')
c.addFunction(x**2-2,'x')
c.addFunction(x**3-2,'x')
c.addFunction(x**4,'x')
c.addFunction(sp.cos(x),'x')
c.addFunction(sp.sin(x),'x')
c.addFunction(sp.tan(x),'x')
#print(c.approxSDer(x**3-2, 3, 'on'))
c.rightsum(x**2-2, 2, 5, 'on')
