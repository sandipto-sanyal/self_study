# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:29:50 2020

@author: sandipto.sanyal
"""
cpdef int test(int x):
    cdef int y = 1
    cdef int i
    for i in range(1, x+1):
        y *= i
    return y