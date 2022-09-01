# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:25:53 2022

@author: jason
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sympy as sp
from sympy import sympify, lambdify
from sympy.abc import x,y,z,t,u,v

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

class Function(object):
    
    def __init__(self):
        self.multivarfunctions = {}
        self.paramfunctions = {}
        self.surfacefunctions = {}
        self.vectorfunctions = {}
        self.variables = {}
        self.step = {}
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
    
    def addVar(self, var):
        self.variables[var] = []
        
    def addRange(self, var, minR, maxR, step):
        self.step[var] = step
        rang = np.arange(minR,maxR+step,step)
        for i in range(len(rang)):
            rang[i] = round(rang[i], int(np.ceil(abs(np.log10(step)))))
        self.variables[var] = rang
    
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
    
    def lambdifyFunction(self, function, variable):
        myfunction=sympify(function)
        if variable == 'x':
            mylambdifiedfunction=lambdify(x,myfunction,'numpy')
        elif variable == 'y':
            mylambdifiedfunction=lambdify(x,myfunction,'numpy')
        return mylambdifiedfunction
    
    def lambdifymultivarFunction(self, function):
        myfunction=sympify(function)
        mylambdifiedfunction=lambdify([x,y],myfunction,'numpy')
        return mylambdifiedfunction
    
    def lambdifysurfaceFunction(self, function):
        myfunction=sympify(function)
        mylambdifiedfunction=lambdify([u,v],myfunction,'numpy')
        return mylambdifiedfunction
    
    def lambdifyparamFunction(self, function):
        myfunction=sympify(function)
        mylambdifiedfunction=lambdify(t,myfunction,'numpy')
        return mylambdifiedfunction
    
    def lambdifyvectorfunction(self, function):
        myfunction=sympify(function)
        mylambdifiedfunction=lambdify([x,y,z],myfunction,'numpy')
        return mylambdifiedfunction
    
    def addmultivarFunction(self, function, xvar, yvar):
        mylambdifiedfunction = self.lambdifymultivarFunction(function)
        x,y = np.meshgrid(self.variables[xvar], self.variables[yvar])
        self.multivarfunctions[function] = [xvar, yvar, x, y, mylambdifiedfunction(x,y)]
    
    def addsurfaceFunction(self, function, uvar, vvar):
        xfunc, yfunc, zfunc = self.lambdifysurfaceFunction(function[0]), self.lambdifysurfaceFunction(function[1]), self.lambdifysurfaceFunction(function[2])
        u,v = np.meshgrid(self.variables[uvar], self.variables[vvar])
        self.surfacefunctions[function] = [uvar, vvar, xfunc(u,v), yfunc(u,v), zfunc(u,v)]
    
    def addparamFunction(self, function, var):
        variable = self.variables[var]
        xfunc,yfunc,zfunc = self.lambdifyparamFunction(function[0]),self.lambdifyparamFunction(function[1]),self.lambdifyparamFunction(function[2])
        self.paramfunctions[function] = [var, self.variables[var], xfunc(variable), yfunc(variable),zfunc(variable)]
    
    def addvectorFunction(self, function, xvar, yvar, zvar):
        x,y,z = np.meshgrid(self.variables[xvar], self.variables[yvar], self.variables[zvar])
        xfunc,yfunc,zfunc = self.lambdifyvectorfunction(function[0]),self.lambdifyvectorfunction(function[1]),self.lambdifyvectorfunction(function[2])
        self.vectorfunctions[function] = [xvar, yvar, zvar, x, y, z, xfunc(x,y,z), yfunc(x,y,z), zfunc(x,y,z)]
    
    def removeFunction(self, function):
        del self.multivarfunctions[function]
        
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
    
    def multivarGraph(self, function):
        x, y, z= list(self.multivarfunctions[function])[2], list(self.multivarfunctions[function])[3], list(self.multivarfunctions[function])[4]
        self.ax.plot_surface(x, y, z, label = self.strFunction(function))
        self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        #plt.legend(ncol = 2, fontsize = 8)
    
    def surfaceGraph(self, function):
        x, y, z= list(self.surfacefunctions[function])[2], list(self.surfacefunctions[function])[3], list(self.surfacefunctions[function])[4]
        self.ax.plot_surface(x, y, z, label = self.strFunction(function))
        self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
    
    def paramGraph(self, function):
        x, y, z = list(self.paramfunctions[function])[2], list(self.paramfunctions[function])[3], list(self.paramfunctions[function])[4]
        self.ax.plot(x, y, z, label = self.strFunction(function))
        self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        #plt.legend(ncol = 2, fontsize = 8)
    
    def vectorGraph(self, function, length = 'off'):
        x,y,z = list(self.vectorfunctions[function])[3], list(self.vectorfunctions[function])[4], list(self.vectorfunctions[function])[5]
        u,v,w = list(self.vectorfunctions[function])[6], list(self.vectorfunctions[function])[7], list(self.vectorfunctions[function])[8]
        if length == 'normalize':
            self.ax.quiver(x,y,z,u,v,w, normalize = True, label = self.strFunction(function))
        elif type(length) == int or type(length) == float:
            u,v,w = length*u, length*v, length*w
            self.ax.quiver(x,y,z,u,v,w, label = self.strFunction(function))
            print('lol')
        else:
            self.ax.quiver(x,y,z,u,v,w, label = self.strFunction(function))
        self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        #plt.legend(ncol = 2, fontsize = 8)
    
    def TNBFrame(self, function, t, graph = 'off'):
        xfunc,yfunc,zfunc = self.lambdifyparamFunction(function[0]),self.lambdifyparamFunction(function[1]),self.lambdifyparamFunction(function[2])
        xp,yp,zp = self.lambdifyparamFunction(sp.diff(function[0])),self.lambdifyparamFunction(sp.diff(function[1])),self.lambdifyparamFunction(sp.diff(function[2]))
        x1, y1, z1 = xfunc(t), yfunc(t), zfunc(t)
        xt, yt, zt = xp(t),yp(t),zp(t)
        magTang = np.sqrt(xt**2 + yt**2 + zt**2)
        xdp,ydp,zdp = self.lambdifyparamFunction(sp.diff(sp.diff(function[0]))),self.lambdifyparamFunction(sp.diff(sp.diff(function[1]))),self.lambdifyparamFunction(sp.diff(sp.diff(function[2])))
        xn, yn, zn = xdp(t),ydp(t),zdp(t)
        magNorm = np.sqrt(xn**2 + yn**2 + zn**2)
        xt, yt, zt = xt/magTang, yt/magTang, zt/magTang
        Tang = [xt,yt,zt]
        xn, yn, zn = xn/magNorm, yn/magNorm, zn/magNorm
        Norm = [xn,yn,zn]
        Tang, Norm = np.array(Tang), np.array(Norm)
        Binorm = np.cross(Tang, Norm)
        if graph == 'on':
            self.ax.quiver(x1, y1, z1, Tang[0], Tang[1], Tang[2], normalize = True, color = 'red', label = 'Tangent Vector')
            self.ax.quiver(x1, y1, z1, Norm[0], Norm[1], Norm[2], normalize = True, color = 'green', label = 'Normal Vector')
            self.ax.quiver(x1, y1, z1, Binorm[0], Binorm[1], Binorm[2], normalize = True, color = 'blue', label = 'Binormal Vector')
            #plt.legend(ncol = 2, fontsize = 8)
            self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        return [Tang, Norm, Binorm]
    
    #def ArcLength(self, function, a, b):
    #    xfunc,yfunc,zfunc = self.lambdifyparamFunction(function[0]),self.lambdifyparamFunction(function[1]),self.lambdifyparamFunction(function[2])
    #    velocity = np.sqrt(xfunc**2 + yfunc**2 + zfunc**2)
    #    return sp.integrate(velocity, (t, a, b))
    
    def Curvature(self, function, t):
        xfunc,yfunc,zfunc = self.lambdifyparamFunction(function[0]),self.lambdifyparamFunction(function[1]),self.lambdifyparamFunction(function[2])
        step = self.step[self.paramfunctions[function][0]]
        xp, yp, zp = self.lambdifyparamFunction(sp.diff(function[0])),self.lambdifyparamFunction(sp.diff(function[1])),self.lambdifyparamFunction(sp.diff(function[2]))
        xdp, ydp, zdp = self.lambdifyparamFunction(sp.diff(sp.diff(function[0]))),self.lambdifyparamFunction(sp.diff(sp.diff(function[1]))),self.lambdifyparamFunction(sp.diff(sp.diff(function[2])))
        xprime, xdprime = np.array([xp,yp,zp]), np.array([xdp,ydp,zdp])
        num = np.cross(xprime, xdprime)
        num = np.sqrt(num[0]**2 + num[1]**2 + num[2]**2)
        denom = (xp**2 + yp**2 + zp**2)**(3/2)
        return num/denom
    
    def AccelerationFrame(self, function, t, graph = 'off'):
        xfunc,yfunc,zfunc = self.lambdifyparamFunction(function[0]),self.lambdifyparamFunction(function[1]),self.lambdifyparamFunction(function[2])
        step = self.step[self.paramfunctions[function][0]]
        x1, y1, z1 = xfunc(t), yfunc(t), zfunc(t)
        xp, yp, zp = self.lambdifyparamFunction(sp.diff(function[0])),self.lambdifyparamFunction(sp.diff(function[1])),self.lambdifyparamFunction(sp.diff(function[2]))
        xdp, ydp, zdp = self.lambdifyparamFunction(sp.diff(sp.diff(function[0]))),self.lambdifyparamFunction(sp.diff(sp.diff(function[1]))),self.lambdifyparamFunction(sp.diff(sp.diff(function[2])))
        xprime, xdprime = np.array([xp,yp,zp]), np.array([xdp,ydp,zdp])
        V = np.sqrt(xp**2 + yp**2 + zp**2)
        AT = (np.dot(xprime, xdprime))/V
        num = np.cross(xprime, xdprime)
        AN = (np.sqrt(num[0]**2 + num[1]**2 + num[2]**2))/V
        tangent = list(self.TNBFrame(function,t))[0]
        normal = list(self.TNBFrame(function,t))[1]
        ATVector = np.array([tangent[0]*AT, tangent[1]*AT, tangent[2]*AT])
        ANVector = np.array([normal[0]*AN, normal[1]*AN,normal[2]*AN])
        if graph == 'on':
            self.ax.quiver(x1, y1, z1, ATVector[0], ATVector[1], ATVector[2], color = 'purple', label = 'Tangent Acceleration Vector')
            self.ax.quiver(x1, y1, z1, ANVector[0], ANVector[1], ANVector[2], color = 'orange', label = 'Normal Acceleration Vector') 
            #plt.legend(ncol = 2, fontsize = 8)
            self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        return ATVector, ANVector
        
    def TangentParam(self, function, t):
        var = self.paramfunctions[function][0]
        rang = self.variables[var]
        xfunc,yfunc,zfunc = self.lambdifyparamFunction(function[0]),self.lambdifyparamFunction(function[1]),self.lambdifyparamFunction(function[2])
        x1, y1, z1 = xfunc(t), yfunc(t), zfunc(t)
        tangent = list(self.TNBFrame(function,t))[0]
        xprime, yprime, zprime = tangent[0], tangent[1], tangent[2]
        self.ax.plot(xprime*rang + x1, yprime*rang + y1, zprime*rang + z1, label = 'Tangent Line at t = ' + str(t))
        self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        #plt.legend(ncol = 2, fontsize = 8)
    
    def NormalParam(self, function, t):
        var = self.paramfunctions[function][0]
        rang = self.variables[var]
        xfunc,yfunc,zfunc = self.lambdifyparamFunction(function[0]),self.lambdifyparamFunction(function[1]),self.lambdifyparamFunction(function[2])
        x1, y1, z1 = xfunc(t), yfunc(t), zfunc(t)
        normal = list(self.TNBFrame(function,t))[1]
        xprime, yprime, zprime = normal[0], normal[1], normal[2]
        self.ax.plot(xprime*rang + x1, yprime*rang + y1, zprime*rang + z1, label = 'Normal Line at t = ' + str(t))
        self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        #plt.legend(ncol = 2, fontsize = 8)
        
    def NormalPlane(self, function, t):
        var = self.paramfunctions[function][0]
        rang = self.variables[var]
        xfunc,yfunc,zfunc = self.lambdifyparamFunction(function[0]),self.lambdifyparamFunction(function[1]),self.lambdifyparamFunction(function[2])
        x1, y1, z1 = xfunc(t), yfunc(t), zfunc(t)
        tangent = list(self.TNBFrame(function,t))[0]
        xprime, yprime, zprime = tangent[0], tangent[1], tangent[2]
        x,y = rang,rang
        x,y = np.meshgrid(x,y)
        eq = -1*((xprime*(x - x1) + yprime*(y - y1))/zprime) + z1
        self.ax.plot_surface(x, y, eq, alpha = 0.5, label = 'Normal Plane')
        self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        #plt.legend(ncol = 2, fontsize = 8)
        
    def OsculatingPlane(self, function, t):
        var = self.paramfunctions[function][0]
        rang = self.variables[var]
        xfunc,yfunc,zfunc = self.lambdifyparamFunction(function[0]),self.lambdifyparamFunction(function[1]),self.lambdifyparamFunction(function[2])
        x1, y1, z1 = xfunc(t), yfunc(t), zfunc(t)
        binorm = list(self.TNBFrame(function,t))[2]
        xprime, yprime, zprime = binorm[0], binorm[1], binorm[2]
        x,y = rang,rang
        x,y = np.meshgrid(x,y)
        eq = -1*((xprime*(x - x1) + yprime*(y - y1))/zprime) + z1
        self.ax.plot_surface(x, y, eq, alpha = 0.5, label = 'Normal Plane')
        self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        #plt.legend(ncol = 2, fontsize = 8)
    
    def xTangent(self, function, x1, y1, graph = 'on'):
        func = self.lambdifymultivarFunction(function)
        x2, y2 = self.variables[(self.multivarfunctions[function])[0]], self.variables[(self.multivarfunctions[function])[1]]
        y3 = np.array([0]*len(y2))
        z2 = func(x2, y2)
        z = func(x1,y1)
        xpartial = self.lambdifymultivarFunction(sp.diff(function, x))
        if graph != 'off':
            self.ax.plot(x2 + x1, y3 + y1, xpartial(x1,y1)*x2 + z, label = 'X Tangent')
            self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        return xpartial(x1,y1)
        
    def yTangent(self, function, x1, y1, graph = 'on'):
        func = self.lambdifymultivarFunction(function)
        x2, y2 = self.variables[(self.multivarfunctions[function])[0]], self.variables[(self.multivarfunctions[function])[1]]
        x3 = np.array([0]*len(x2))
        z2 = func(x2, y2)
        z = func(x1,y1)
        ypartial = self.lambdifymultivarFunction(sp.diff(function, y))
        if graph != 'off':
            self.ax.plot(x3 + x1, y2 + y1, ypartial(x1,y1)*x2 + z, label = 'Y Tangent')
            self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        return ypartial(x1, y1)
    
    def xyTangentPlane(self, function, x, y):
        xvar, yvar = self.multivarfunctions[function][0], self.multivarfunctions[function][1]
        xrang, yrang = self.variables[xvar], self.variables[yvar]
        x1,y1 = np.meshgrid(xrang, yrang)
        z = self.lambdifymultivarFunction(function)(x,y)
        xpartial, ypartial = self.xTangent(function, x, y, 'off'), self.yTangent(function, x, y, 'off')
        eq = ((xpartial*(x1 - x) + ypartial*(y1 - y))) + z
        self.ax.plot_surface(x1, y1, eq)
    
    def DirectionalDer(self, function, x, y, u, v):
        xpartial, ypartial = self.xTangent(function, x, y, 'off'), self.yTangent(function, x, y, 'off')
        grad = np.array([xpartial, ypartial])
        vector = np.array([u, v])
        return np.dot(grad, vector)
    
    def DoubleIntegral(self, function, xa, xb, ya, yb, typ, step1 = 'same'):
        def rect_prism(x, y, z, step):
            xdis, ydis, zdis = x[1] - x[0], y[1] - y[0], z[1] - z[0]
            t = np.arange(0, 1+step, step)
            self.ax.plot(xdis*t + x[0], 0*t+y[0],0*t+z[0], color = 'red')
            self.ax.plot(0*t+x[1], ydis*t + y[0],0*t+z[0], color = 'red')
            self.ax.plot(xdis*t + x[0], 0*t+y[1],0*t+z[0], color = 'red')
            self.ax.plot(0*t+x[0], ydis*t + y[0],0*t+z[0], color = 'red')
            self.ax.plot(0*t+x[0], 0*t+y[0],zdis*t + z[0], color = 'red')
            self.ax.plot(0*t+x[1], 0*t+y[0],zdis*t + z[0], color = 'red')
            self.ax.plot(0*t+x[0], 0*t+y[1],zdis*t + z[0], color = 'red')
            self.ax.plot(0*t+x[1], 0*t+y[1],zdis*t + z[0], color = 'red')
            self.ax.plot(xdis*t + x[0], 0*t+y[0],0*t+z[1], color = 'red')
            self.ax.plot(0*t+x[1], ydis*t + y[0],0*t+z[1], color = 'red')
            self.ax.plot(xdis*t + x[0], 0*t+y[1],0*t+z[1], color = 'red')
            self.ax.plot(0*t+x[0], ydis*t + y[0],0*t+z[1], color = 'red')
        func = self.lambdifymultivarFunction(function)
        integral = 0
        if step1 == 'same':
            step = (self.step[self.multivarfunctions[function][0]] + self.step[self.multivarfunctions[function][1]])/2
        else:
            step = step1
        x, y = np.arange(xa, xb+step, step), np.arange(ya, yb+step, step)
        if typ == 'left':
            for i in range(len(x)-1):
                for j in range(len(y)-1):
                    x_range, y_range, z_range = np.array([x[i], x[i+1]]), np.array([y[j], y[j+1]]), np.array([0, func(x[i], y[j])])
                    rect_prism(x_range, y_range, z_range, step)
                    integral += func(x[i], y[j])*xdis*ydis
        if typ == 'right':
            for i in range(len(x)-1):
                for j in range(len(y)-1):
                    x_range, y_range, z_range = np.array([x[i], x[i+1]]), np.array([y[j], y[j+1]]), np.array([0, func(x[i+1], y[j+1])])
                    rect_prism(x_range, y_range, z_range, step)
                    integral += func(x[i+1], y[j+1])*xdis*ydis
        return integral
    
    def LevelCurves(self, function, z, threeD = 'off'):
        xvar, yvar = self.multivarfunctions[function][0], self.multivarfunctions[function][1]
        xrang, yrang = self.variables[xvar], self.variables[yvar]
        y1  = sp.solve(function-z, y)
        for i in range(len(y1)):
            yfunc = self.lambdifyFunction(y1[i], 'x')
            if threeD == 'on':
                self.ax.plot(xrang, yfunc(xrang), [z]*len(xrang))
            else: 
                self.ax.plot(xrang, yfunc(xrang), [0]*len(xrang))
            self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
    
    def LevelCurveIterator(self, function, zmin, zmax, step=1, threeD = 'off'):
        iterations = int((zmax - zmin)/step)
        zrange = []
        for i in range(iterations+1):
            zrange.append(zmin + step*i)
        for item in zrange:
            if threeD == 'on':
                self.LevelCurves(function, item, 'on')
            else:
                self.LevelCurves(function, item)
            self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
    
    def xyzPlaneGraph(self, function, plane, var):
        xvar, yvar = self.multivarfunctions[function][0], self.multivarfunctions[function][1]
        xrang, yrang = self.variables[xvar], self.variables[yvar]
        func = self.lambdifymultivarFunction(function)
        if plane == 'xy' or plane == 'yx':
            self.LevelCurves(function, var, 'on')
        elif plane == 'xz' or plane == 'zx':
            y = 0
            zfunc = self.lambdifymultivarFunction(function - var)
            self.ax.plot(xrang, [var]*len(xrang), zfunc(xrang,0))
        elif plane == 'yz' or plane == 'zy':
            x = 0 
            zfunc = self.lambdifymultivarFunction(function - var)
            self.ax.plot([var]*len(yrang), yrang, zfunc(0,yrang))
        
    def xyPlaneGraph(self, function, var):
        xvar, yvar = self.multivarfunctions[function][0], self.multivarfunctions[function][1]
        xrang, yrang = self.variables[xvar], self.variables[yvar]
        y1  = sp.solve(function-var, y)
        for i in range(len(y1)):
            yfunc = self.lambdifyFunction(y1[i], 'x')
            self.ax.plot(xrang, yfunc(xrang), [var]*len(xrang))

    def uTangent(self, function, u1, v1, graph = 'on'):
        xfunc, yfunc, zfunc = self.lambdifysurfaceFunction(function[0]), self.lambdifysurfaceFunction(function[1]), self.lambdifysurfaceFunction(function[2])
        xpfunc, ypfunc, zpfunc = self.lambdifysurfaceFunction(sp.diff(function[0], u)), self.lambdifysurfaceFunction(sp.diff(function[1], u)), self.lambdifysurfaceFunction(sp.diff(function[2], u))
        urang, vrang = self.variables[self.surfacefunctions[function][0]], self.variables[self.surfacefunctions[function][1]]
        v2 = np.array([0]*len(vrang))
        if graph != 'off':
            self.ax.plot(xpfunc(u1,v1)*urang + xfunc(u1,v1), ypfunc(u1,v1)*urang + yfunc(u1,v1), zpfunc(u1,v1)*urang + zfunc(u1,v1))
            self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        return np.array([xpfunc(u1,v1), ypfunc(u1,v1), zpfunc(u1,v1)])
    
    def vTangent(self, function, u1, v1, graph = 'on'):
        xfunc, yfunc, zfunc = self.lambdifysurfaceFunction(function[0]), self.lambdifysurfaceFunction(function[1]), self.lambdifysurfaceFunction(function[2])
        xpfunc, ypfunc, zpfunc = self.lambdifysurfaceFunction(sp.diff(function[0], v)), self.lambdifysurfaceFunction(sp.diff(function[1], v)), self.lambdifysurfaceFunction(sp.diff(function[2], v))
        urang, vrang = self.variables[self.surfacefunctions[function][0]], self.variables[self.surfacefunctions[function][1]]
        v2 = np.array([0]*len(vrang))
        if graph != 'off':
            self.ax.plot(xpfunc(u1,v1)*vrang + xfunc(u1,v1), ypfunc(u1,v1)*vrang + yfunc(u1,v1), zpfunc(u1,v1)*vrang + zfunc(u1,v1))
            self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
        return np.array([xpfunc(u1,v1), ypfunc(u1,v1), zpfunc(u1,v1)])
    
    def uvTangentPlane(self, function, u, v):
        xfunc, yfunc, zfunc = self.lambdifysurfaceFunction(function[0]), self.lambdifysurfaceFunction(function[1]), self.lambdifysurfaceFunction(function[2])
        urang, vrang = self.variables[self.surfacefunctions[function][0]], self.variables[self.surfacefunctions[function][1]]
        urang, vrang = np.meshgrid(urang, vrang)
        uprime, vprime = self.uTangent(function, u, v, 'off'), self.vTangent(function, u, v, 'off')
        normal = np.cross(uprime, vprime)
        eq = -1*((normal[0]*(urang - xfunc(u,v)) + normal[1]*(vrang - yfunc(u,v)))/(normal[2])) + zfunc(u,v)
        self.ax.plot_surface(urang, vrang, eq)
        self.ax.set_box_aspect([ub - lb for lb, ub in (getattr(self.ax, f'get_{a}lim')() for a in 'xyz')])
    
    def __str__(self):
        return str(self.multivarfunctions), str(self.paramfunctions)
