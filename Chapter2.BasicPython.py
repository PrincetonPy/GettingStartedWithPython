# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Chapter 2. Basic Python

# <markdowncell>

# This section moves quickly. I'm assuming that everyone speaks at least one programming language, and so this chapter gives a lightning intro to syntax in Python. The sections are subheaded, but they really overlap quite a lot, so they're there more as a page reference...

# <headingcell level=3>

# Variables and Arithmetic

# <codecell>

print("Is this thing on ?")
print 'Yeah, I think so.'

# <markdowncell>

# In standard programming language tradition, we begin with a "Hello world" equivalent. We can immediately see a few things :
# 
# 1. Standard in / out ( writing to screen, for instance ) is available without invoking any libraries. That is, the Python Standard Library comes with some useful things - w00t.
# 
# 2. Strings are delimited by `""`, but can also use `''`. This is useful because you can now use one set of quotes inside another, and it'll still be one big string.
# 
# 3. The `print` function seems to be happy with and without brackets. This is only true in Py2. Py3 requires brackets, and because `print` is a function that takes arguments like any other, we'll *try* keep the brackets as good practice - but I usually fail here...

# <codecell>

s = "This is a string."
print(s)
print(type(s))

s = 42
print(s)
print(type(s))

# <rawcell>

# Variables don't need to be given a type, as Python is *dynamically-typed*. That means if I wanted to reuse `s` as an integer, Python would have no issue with that.

# <codecell>

print(s * 2) # Multiplication
print(s + 7) # This one's really hard

# <markdowncell>

# Single-line comments use `#`. Arithmetic uses the standard operators : `+`, `-`, `*`, `/`. You can take powers using `**`. Python also allows C-like `+=` syntax :

# <codecell>

s += 2**3
print s

# <markdowncell>

# Note that, two cells up, we called `s*2` and `s+7`, but never used `=`, so we never changed `s`. In the last cell, however, we used `+=`. 
# 
# That statement is equivalent to saying `s = s + 2**3`, it's just shorthand. Also works with `-=`, `*=`, `/=`, `**=`

# <codecell>

print(s == 42)
print(s == 50)
print(s > 10)

# <markdowncell>

# The `==` operator is the *comparison* operator. Here, we also see Python's syntax for logical statements : `True` and `False`. As with any programming syntax, capitalisation is important. In Python, `1` is also `True`, and `0` is also `False`.

# <codecell>

x = "Blah"
print(x + x)
print(len(x))

# <headingcell level=3>

# Lists

# <markdowncell>

# Strings can be concatenated using the `+` operator. The `len()` function returns the length of a string. This also works for lists :

# <codecell>

mylist = [1, 2.41341]
mylist.append("We can mix types !")

print(mylist)
print("Length is " + str(len(mylist))) # note how we TYPECAST an integer ( returned by len() ) into a string
print(type(mylist))

# <markdowncell>

# Here, we typecast the integer `len(mylist)` into a string by simply calling `str` on it. We can typecast to any valid Python type : `float`, `int`, `str`, `complex`, ...
# 
# Python accesses elements in lists from 0, not from 1 as in Matlab or R. This will be familiar to C users.

# <codecell>

print(mylist[0])
print(mylist[1])
print(mylist[2])

# <headingcell level=3>

# Control Structures

# <markdowncell>

# For loops in Python are really cool. Python objects like lists are *iterables*. That is, we can directly iterate over them :

# <codecell>

for i in mylist :
    print i

# <markdowncell>

# *Note the indentation* - this is really important. Loops in Python don't get delimited by brackets like in C or R. Each block gets its own indentation. Typically, people use tabs, but you can use any amount of whitespace you want *as long as you are consistent*. To end the loop, simply unindent. We'll see that in a few lines.
# 
# Here, the keyword `in` individually returns members of `mylist`. It can also be used to check whether something is in a container :

# <codecell>

print (1 in mylist)
print (2 in mylist)

# <markdowncell>

# If you wanted to loop by number of the list, we can use `range()`, which, in its simplest ( single-argument ) form, returns a list from 0 to that element minus 1.

# <codecell>

print(range(5))

for i in range(len(mylist)) :
    print i, mylist[i]

# <markdowncell>

# A quick way to do this is the `enumerate` function :

# <codecell>

for index, value in enumerate(mylist) :
    print("Element number " + str(index) + " in the list has the value " + str(value))

# <markdowncell>

# Great. What about while loops and if statements ?

# <codecell>

x = 10

while x > 0 :
    
    if x != 1 :
        print(str(x) + " bottles of beer on the wall,")
        
    elif x == 1 :
        print(str(x) + " bottle of beer on the wall,")
        
    else : 
        print("This will never actually happen.")
        
    x -= 1
    
print "All the beer is gone :("

# <markdowncell>

# Notice how the contents of the while loop are indented, and then code that is outside the loop continues unindented below. Here's a nested loop to clarify :

# <codecell>

for i in range(1, 5) :
    blah = 0
    
    for j in range(i) :
        blah += j
        
    print "i = " + str(i), blah

# <markdowncell>

# Here, we used `range()` with two arguments. It generates a list from the first argument to the second argument minus 1. Also, note that we can feed the `print` statement several things to print, separated by a comma. Again, we can use brackets if we want for `print`.

# <headingcell level=3>

# Interacting Between Different Variable Types

# <markdowncell>

# Beware of integer division. Unlike C, Python *uptypes* cross-type operations. Unlike R, Python does not uptype integer operations.

# <codecell>

myint = 2
myfloat = 3.14
print type(myint), type(myfloat)

# <codecell>

# Multiplying an int with a float gives a float, we've UPTYPED
print myint * myfloat
print type(myint * myfloat)

# <codecell>

# But operations between SAME type gives the same type :
print 7 / 3
print type(7/3)

# <codecell>

# Quick hack with ints to floats - there's no need to typecast, just give it a float 
print 7. / 3
print type(7 / 3.0)

# <headingcell level=3>

# More Lists : Accessing Elements

# <markdowncell>

# Let's go back to lists. They're a type of generic, ordered container; their elements can be access in several ways.

# <codecell>

# Create a list of integers 0, 1, 2, 3, 4
A = range(5);
print A

# <codecell>

# Let's replace the middle element
A[2] = "Naaaaah"
print A

# <codecell>

# What are the middle three elements ? Let's use the : operator
# Like range(), it creates a list of integers
# 1:4 will give us 1, 2, 3 because we stop at n-1, like with range()
print A[1:4]

# <codecell>

# So what's the point of both range() and the : operator ?
# We don't need to give it a start or an end with :
print A[:2]
print A[2:]

# <codecell>

# Can we access the last element ? What about the last two ?
print A[len(A)-2:]
print A[-2:]

# <codecell>

# Earlier, we saw that range() can take two arguments : range(start, finish) 
# It can actually take a third : range(start, finish, stride)
print range(0, 10, 2)

# <codecell>

# Similarly, the : operator can also do this
print A[0:5:2]
# Here, it will give us elements 0, 2, 4.

# <codecell>

# Again, what if I don't want to explicitly remember the size of the list ?
print A[::2]
# This will simply go from start to finish with a stride of 2

# <codecell>

# And this one, from the second element to finish, with a stride of 2
print A[1::2]

# <codecell>

# So, uh... Reverse ?
print A[::-1]

# <codecell>

# List arithmetic is fun, but don't get too attached : you may not be using many lists in the future...
print A + A
print A * 4

# <headingcell level=3>

# Dictionaries

# <markdowncell>

# Let's take a very brief look at *dictionaries*. These are unordered containers that you can use to pair elements in, similar to a `std::map` if you're a C++ coder. 

# <codecell>

pythonSkillz = { "Quentin" : 37651765, "Paul" : 0.2 }
print pythonSkillz

# <codecell>

# Dictionaries associate keys with values
print pythonSkillz.keys()
print pythonSkillz.values()

# <codecell>

# You can access them through their keys
print pythonSkillz["Quentin"]

# <markdowncell>

# There are a couple of other built-in containers, like *tuples* and *sets*. I won't go into them here, plainly because I have to use them so rarely that it's not worth the time during the session. If you want to read up : `http://docs.python.org/2/tutorial/datastructures.html`

# <headingcell level=3>

# List Comprehension and Inlines

# <markdowncell>

# A couple of cool final tricks :

# <codecell>

# Let's build a list of elements 1^2, 2^2, 3^2, ..., 10^2
x = range(1, 11)
print [i**2 for i in x]

# <codecell>

# We can inline if statements too
print "Trust me, I know what I'm doing." if pythonSkillz["Paul"] > 0 else "Whaaaaaaa"

# <markdowncell>

# That concludes the syntactical introduction to Python. Next : functions.

# <headingcell level=3>

# Functions

# <codecell>

# Fibonacci numbers
# OH NO RECURSION

def fib(n) :
    if n == 0 :
        return 0
    elif n == 1 :
        return 1
    else :
        return fib(n-1) + fib(n-2)

# <codecell>

# Testing :
for i in range(10): 
    print fib(i)

# <markdowncell>

# Looks good. We've just defined a function that takes one argument, `n`, and returns something based on what `n` is. The Fibonacci function is quite particular because it calls itself ( recursion ), but it's a small, fun example, so why not.

# <headingcell level=3>

# Better Printing

# <markdowncell>

# Earlier, I used the `+` operator to *concatenate* a number into a string. To do that, I had to typecast it. Not great. This is the better way to do it :

# <codecell>

def printFib(i) :
    print "The %dth number of the Fibonnaci sequence is %d." % (i, fib(i))

# <codecell>

printFib(20)

# <markdowncell>

# Here, `%d` is a format code for *integer*. `%f` is for floating point numbers ( floats ), and `%s` is for strings. The sneaky one to know is `%r`, which just takes any type.
# 
# Note how, to pass more than one thing in, we had to put it into round brackets. This is a *tuple*, we mentioned it briefly in the last notebook. It's basically just an immutable list. String formatting like this takes tuples. 

# <codecell>

# I stole this one from Learn Python The Hard Way ( highly recommended ) :
formatstring = "%r %r %r %r"
print formatstring % (formatstring, formatstring, formatstring, formatstring)

# <markdowncell>

# Also worth knowing are `\n` and `\t` : the newline and tab characters, respectively.

# <codecell>

# Written on-the-fly, because I got mad skills
print "This is a haiku\n\tI'm crappy at poetry\n\t\tWait, this really worked"

# <headingcell level=3>

# File IO

# <markdowncell>

# A very, *very* quick look at file IO, because there are packages that can do a better job.

# <codecell>

myfile = open("README.MD", "r")
for line in myfile :
    print line
    
# There are other options instead of looping over each line. You can instead use myfile.read().
# Writing : you can dump a variable using myfile.write() after having opened it in "w" mode.

# There are many other ways to read and write files, including ways to read and write CSV directly.

# <markdowncell>

# OK, with that - onto the Scientific Python Stack.

