# Developer Interview Questions

Common Junior dev Questions curated from the internet.<br>
*Disclaimer* I'm not in HR.<br>
Sources
1. [interview-questions-for-graduatejunior-software-developers](https://ilovefoobar.wordpress.com/2012/12/15/interview-questions-for-graduatejunior-software-developers/)
2. [https://www.fullstack.cafe/](https://www.fullstack.cafe/)

<br>

## Questions
<details><summary><b>Tell us  about yourself.</b></summary>
<p>
one of the best strategy is to focus on the employer and your fit for this job. No body wants to know about your 10 cats.
</p>
</details>
<details><summary><b>Other than study and programming, what you like to do during your free time.</b></summary>

> show you have a life but keep it relevant to the job

</details>

<details><summary><b>What is difference between overloading and overriding in OOP.</b></summary>
<p>

> * Overloading* occurs when two or more methods in one class have the same method name but different parameters.<br>
> * Overriding* means having two methods with the same method name and parameters (i.e., method signature). One of the methods is in the parent class and the other is in the child class. Overriding allows a child class to provide a specific implementation of a method that is already provided its parent class.

</p>
</details>

<details><summary><b>Tell us about your experience while working in team.</b></summary>
<p>

> Aim of this question is to find out if you're a team play. Don't imply that without you the team wouldn't make it also be careful not to come across as the weakest link in the team. Mention your achievemnts personal and also as a team.

</p>
</details>
<details><summary><b>How do you manage conflicts in a group assignments. </b></summary>

> aim is to show you're mature and professional in handling conflict.

</details>


<details><summary><b> Difference btwn Localstorage, sessionstorage and cookie </b></summary>

> localStorage: stores data with no expiration date, and gets cleared only through JavaScript, or clearing the Browser Cache / Locally Stored Data

> sessionStorage: similar to localStorage but expires when the browser closed (not the tab).

> Cookie: stores data that has to be sent back to the server with subsequent requests. Its expiration varies based on the type and the expiration duration can be set from either server-side or client-side (normally from server-side).
  
</details>

<details><summary><b>Write a sql query to join two tables in database.</b></summary>
<p>

> * (INNER) JOIN: Returns records that have matching values in both tables<br><br>
`SELECT column_name(s)
FROM table1
INNER JOIN table2
ON table1.column_name = table2.column_name;`<br><br>
> * LEFT (OUTER) JOIN: Returns all records from the left table, and the matched records from the right table<br><br>
`SELECT column_name(s)
FROM table1
LEFT JOIN table2
ON table1.column_name = table2.column_name;`<br><br>
> * RIGHT (OUTER) JOIN: Returns all records from the right table, and the matched records from the left table <br><br>
`SELECT column_name(s)
FROM table1
RIGHT JOIN table2
ON table1.column_name = table2.column_name;`<br><br>
> * FULL (OUTER) JOIN: Returns all records when there is a match in either left or right table <br><br>
`SELECT column_name(s)
FROM table1
FULL OUTER JOIN table2
ON table1.column_name = table2.column_name
WHERE condition;`<br><br>

</p>
</details>
<details><summary><b>Imagine you have two array a = [1,2,3,4,5] and b =[3,2,9,3,7], write a program to find out common elements in both array.</b></summary>


```
a = [1, 2, 3, 4, 5]
b = [3, 2, 9, 3, 7]
temp = []
for i in range(len(b)):
    if a[i] in b:
        temp.append(a[i])

print(temp)

```


</details>


<details><summary><b>( Related to question above.) Can you write this without using for loop? </b></summary>

```

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        print(a_set & b_set)
    else:
        print("No common elements")
a = [1, 2, 3, 4, 5]
b = [3, 2, 9, 3, 7]
common_member(a, b)
```


   </p>
</details>


<details><summary><b> If i sort those arrays will it make any difference in your code? Can you write better code if arrays are sorted? </b></summary>
   
   >	Time complexity will be same.		
   
</details>

<details><summary><b> What is different between ArrayList and Set.</b></summary>

> List is a type of ordered collection that maintains the elements in insertion order while Set is a type of unordered collection so elements are not maintained any order.

> List allows duplicates while Set doesn't allow duplicate elements . All the elements of a Set should be unique if you try to insert the duplicate element in Set it would replace the existing value.

> List permits any number of null values in its collection while Set permits only one null value in its collection.

> New methods are defined inside List interface . But, no new methods are defined inside Set interface, so we have to use Collection interface methods only with Set subclasses .

> List can be inserted in in both forward direction and backward direction using Listiterator while Set can be traversed only in forward direction with the help of iterator 
 
</details>

<details><summary><b> Write sql query to find out total number of item sold 
for certain product and list them in descending order. </b></summary>

```
SELECT ProductID, count(*) AS NumSales FROM Orders GROUP BY ProductID DESC;
```

</details>


<details><summary><b>What is use of index in database? Give us example of columns that should be indexed. </b></summary>

>  Indexes are used to quickly locate data without having to search every row in a database table every time a database table is accessed. You can use a combination of columns. you can index UPPER(LastName)

</details>


<details><summary><b> What is use of foreign key in database? </b></summary>
 
 > A foreign key is a column or group of columns in a relational database table that provides a link between data in two tables. It acts as a cross-reference between tables because it references the primary key of another table, thereby establishing a link between them

</details>


<details><summary><b> What is MVC pattern? </b></summary>

> an architectural pattern commonly used for developing user interfaces that divides an application into three interconnected parts. This is done to separate internal representations of information from the ways information is presented to and accepted from the user

</details>


<details><summary><b>Have you heard of any design pattern? Please name and explain couple of them. </b></summary>
 
 > [https://sourcemaking.com/design_patterns](https://sourcemaking.com/design_patterns)

</details>


<details><summary><b> What is data-structure? </b></summary>

> Data structure availability may vary by programming languages. Commonly available data structures are:
   * list,
   * arrays,
   * stack,
   * queues,
   * graph,
   * tree etc

</details>


<details><summary><b> What is algorithm?</b></summary>
  
  > Algorithm is a step by step procedure, which defines a set of instructions to be executed in certain order to get the desired output.

</details>


<details><summary><b>What is linear searching? </b></summary>

> Linear search or sequential search is a method for finding a target value within a list. It sequentially checks each element of the list for the target value until a match is found or until all the elements have been searched. Linear search runs in at worst linear time and makes at most n comparisons, where n is the length of the list. 

</details>


<details><summary><b>  What is a graph? </b></summary>
 
  > A graph is a pictorial representation of a set of objects where some pairs of objects are connected by links. The interconnected objects are represented by points termed as vertices, and the links that connect the vertices are called edges.

</details>


<details><summary><b> Why we need to do algorithm analysis?  </b></summary>

> A problem can be solved in more than one ways. So, many solution algorithms can be derived for a given problem. We analyze available algorithms to find and implement the best suitable algorithm.

> An algorithm are generally analyzed on two factors − time and space. That is, how much execution time and how much extra space required by the algorithm.

</details>



<details><summary><b>What is asymptotic analysis of an algorithm?  </b></summary>

> Asymptotic analysis of an algorithm, refers to defining the mathematical boundation/framing of its run-time performance. Using asymptotic analysis, we can very well conclude the best case, average case and worst case scenario of an algorithm.

</details>



<details><summary><b>What is linear data structure and what are common operations to perform on it?  </b></summary>


> A linear data-structure has sequentially arranged data items. The next item can be located in the next memory address. It is stored and accessed in a sequential manner. Array and list are example of linear data structure.

The following operations are commonly performed on any data-structure:

    Insertion − adding a data item
    Deletion − removing a data item
    Traversal − accessing and/or printing all data items
    Searching − finding a particular data item
    Sorting − arranging data items in a pre-defined sequence


</details>



<details><summary><b>What examples of greedy algorithms do you know? </b></summary>


> The below given problems find their solution using greedy algorithm approach:

    Travelling Salesman Problem
    Prim's Minimal Spanning Tree Algorithm
    Kruskal's Minimal Spanning Tree Algorithm
    Dijkstra's Minimal Spanning Tree Algorithm
    Graph - Map Coloring
    Graph - Vertex Cover
    Knapsack Problem
    Job Scheduling Problem


</details>


<details><summary><b> What are some examples of divide and conquer algorithms? </b></summary>

> The below given problems find their solution using divide and conquer algorithm approach:

    Merge Sort
    Quick Sort
    Binary Search
    Strassen's Matrix Multiplication
    Closest pair (points)


</details>


<details><summary><b> What are some examples of dynamic programming algorithms? </b></summary>
   

> The below given problems find their solution using divide and conquer algorithm approach:

    Fibonacci number series
    Knapsack problem
    Tower of Hanoi
    All pair shortest path by Floyd-Warshall
    Shortest path by Dijkstra
    Project scheduling


</details>




<details><summary><b>Why do we use stacks? </b></summary>
 
> In data-structure, stack is an Abstract Data Type (ADT) used to store and retrieve values in Last In First Out (LIFO) method.

> Stacks follows LIFO method and addition and retrieval of a data item takes only Ο(n) time. Stacks are used where we need to access data in the reverse order or their arrival. Stacks are used commonly in recursive function calls, expression parsing, depth first traversal of graphs etc.

> The below operations can be performed on a stack:

    push() − adds an item to stack
    pop() − removes the top stack item
    peek() − gives value of top item without removing it
    isempty() − checks if stack is empty
    isfull() − checks if stack is full


</details>




<details><summary><b> Why do we use queues?  </b></summary>

> Queue is an abstract data structure (ADS), somewhat similar to stack. In contrast to stack, queue is opened at both end. One end is always used to insert data (enqueue) and the other is used to remove data (dequeue). Queue follows First-In-First-Out (FIFO) methodology, i.e., the data item stored first will be accessed first.

> As queues follows FIFO method, they are used when we need to work on data-items in exact sequence of their arrival. Every operating system maintains queues of various processes. Priority queues and breadth first traversal of graphs are some examples of queues.

> The below operations can be performed on a queue:

    enqueue() − adds an item to rear of the queue
    dequeue() − removes the item from front of the queue
    peek() − gives value of front item without removing it
    isempty() − checks if stack is empty
    isfull() − checks if stack is full


</details>



<details><summary><b> What is the difference between Linear Search and Binary Search? </b></summary>
  

    A linear search looks down a list, one item at a time, without jumping. In complexity terms this is an O(n) search - the time taken to search the list gets bigger at the same rate as the list does.

    A binary search is when you start with the middle of a sorted list, and see whether that's greater than or less than the value you're looking for, which determines whether the value is in the first or second half of the list. Jump to the half way through the sublist, and compare again etc. In complexity terms this is an O(log n) search - the number of search operations grows more slowly than the list does, because you're halving the "search space" with each operation.

Comparing the two:

    Binary search requires the input data to be sorted; linear search doesn't
    Binary search requires an ordering comparison; linear search only requires equality comparisons
    Binary search has complexity O(log n); linear search has complexity O(n)
    Binary search requires random access to the data; linear search only requires sequential access (this can be very important - it means a linear search can stream data of arbitrary size)


</details>


<details><summary><b> What is an average case complexity of Bubble Sort? </b></summary>
   
> Bubble sort, sometimes referred to as sinking sort, is a simple sorting algorithm that repeatedly steps through the list to be sorted, compares each pair of adjacent items and swaps them if they are in the wrong order. The pass through the list is repeated until no swaps are needed, which indicates that the list is sorted.

> Bubble sort has a worst-case and average complexity of О(n2), where n is the number of items being sorted. Most practical sorting algorithms have substantially better worst-case or average complexity, often O(n log n). Therefore, bubble sort is not a practical sorting algorithm.

</details>


<details><summary><b> What is Selection Sort?  </b></summary>
   
> Selection sort is in-place sorting technique. It divides the data set into two sub-lists: sorted and unsorted. Then it selects the minimum element from unsorted sub-list and places it into the sorted list. This iterates unless all the elements from unsorted sub-list are consumed into sorted sub-list.

</details>

#### Java OOP

<details><summary><b> What is JVM? Why is Java called the “Platform Independent Programming Language”? </b></summary>
  
> A Java virtual machine (JVM) is a process virtual machine that can execute Java bytecode. Each Java source file is compiled into a bytecode file, which is executed by the JVM. Java was designed to allow application programs to be built that could be run on any platform, without having to be rewritten or recompiled by the programmer for each separate platform. A Java virtual machine makes this possible, because it is aware of the specific instruction lengths and other particularities of the underlying hardware platform.

</details>

<details><summary><b> What is the Difference between JDK and JRE? </b></summary>

> The Java Runtime Environment (JRE) is basically the Java Virtual Machine (JVM) where your Java programs are being executed. It also includes browser plugins for applet execution. The Java Development Kit (JDK) is the full featured Software Development Kit for Java, including the JRE, the compilers and tools (like JavaDoc, and Java Debugger), in order for a user to develop, compile and execute Java applications.
  
</details>

<details><summary><b> What are the two types of Exceptions in Java? Which are the differences between them? </b></summary>
  
> Java has two types of exceptions: checked exceptions and unchecked exceptions. Unchecked exceptions do not need to be declared in a method or a constructor’s throws clause, if they can be thrown by the execution of the method or the constructor, and propagate outside the method or constructor boundary. On the other hand, checked exceptions must be declared in a method or a constructor’s throws clause. 

</details>


<details><summary><b> What is the difference between an Applet and a Java Application? </b></summary>

> Applets are executed within a java enabled browser, but a Java application is a standalone Java program that can be executed outside of a browser. However, they both require the existence of a Java Virtual Machine (JVM). Furthermore, a Java application requires a main method with a specific signature, in order to start its execution. Java applets don’t need such a method to start their execution. Finally, Java applets typically use a restrictive security policy, while Java applications usually use more relaxed security policies.
  
</details>


<details><summary><b>  Let's talk Swing. What is the difference between a Choice and a List? </b></summary>

> A Choice is displayed in a compact form that must be pulled down, in order for a user to be able to see the list of all available choices. Only one item may be selected from a Choice. A List may be displayed in such a way that several List items are visible. A List supports the selection of one or more List items.

</details>


<details><summary><b> What is a JSP Page? </b></summary>
  
> A Java Server Page (JSP) is a text document that contains two types of text: static data and JSP elements. Static data can be expressed in any text-based format, such as HTML or XML. JSP is a technology that mixes static content with dynamically-generated content.

</details>


<details><summary><b>  What is a Servlet? </b></summary>
 
> The servlet is a Java programming language class used to process client requests and generate dynamic web content. Servlets are mostly used to process or store data submitted by an HTML form, provide dynamic content and manage state information that does not exist in the stateless HTTP protocol.

</details>


<details><summary><b>  What does the “static” keyword mean? Can you override private or static method in Java?  </b></summary>
 
> The static keyword denotes that a member variable or method can be accessed, without requiring an instantiation of the class to which it belongs. A user cannot override static methods in Java, because method overriding is based upon dynamic binding at runtime and static methods are statically binded at compile time. A static method is not associated with any instance of a class so the concept is not applicable.

</details>


<details><summary><b> What are the Data Types supported by Java? What is Autoboxing and Unboxing?  </b></summary>
 
> The eight primitive data types supported by the Java programming language are:

    byte
    short
    int
    long
    float
    double
    boolean
    char

</details>


<details><summary><b> What is Function Overriding and Overloading in Java? </b></summary>
 
> Method overloading in Java occurs when two or more methods in the same class have the exact same name, but different parameters. On the other hand, method overriding is defined as the case when a child class redefines the same method as a parent class. Overridden methods must have the same name, argument list, and return type. The overriding method may not limit the access of the method it overrides.

</details>


<details><summary><b>  What is the difference between an Interface and an Abstract class? </b></summary>

> Java provides and supports the creation both of abstract classes and interfaces. Both implementations share some common characteristics, but they differ in the following features:

    All methods in an interface are implicitly abstract. On the other hand, an abstract class may contain both abstract and non-abstract methods.
    A class may implement a number of Interfaces, but can extend only one abstract class.
    In order for a class to implement an interface, it must implement all its declared methods. However, a class may not implement all declared methods of an abstract class. Though, in this case, the sub-class must also be declared as abstract.
    Abstract classes can implement interfaces without even providing the implementation of interface methods.
    Variables declared in a Java interface is by default final. An abstract class may contain non-final variables.
    Members of a Java interface are public by default. A member of an abstract class can either be private, protected or public.
    An interface is absolutely abstract and cannot be instantiated. An abstract class also cannot be instantiated, but can be invoked if it contains a main method.

</details>


<details><summary><b> What are pass by reference and pass by value?  </b></summary>

> When an object is passed by value, this means that a copy of the object is passed. Thus, even if changes are made to that object, it doesn’t affect the original value. When an object is passed by reference, this means that the actual object is not passed, rather a reference of the object is passed. Thus, any changes made by the external method, are also reflected in all places.

</details>


<details><summary><b>  What is the difference between processes and threads?  </b></summary>

> A process is an execution of a program, while a Thread is a single execution sequence within a process. A process can contain multiple threads. A Thread is sometimes called a lightweight process.

</details>


<details><summary><b> What are the basic interfaces of Java Collections Framework? </b></summary>
 
> Java Collections Framework provides a well designed set of interfaces and classes that support operations on a collections of objects. The most basic interfaces that reside in the Java Collections Framework are:

    Collection, which represents a group of objects known as its elements.
    Set, which is a collection that cannot contain duplicate elements.
    List, which is an ordered collection and can contain duplicate elements.
    Map, which is an object that maps keys to values and cannot contain duplicate keys.

  
</details>


<details><summary><b> What is an Iterator? </b></summary>

> The Iterator interface provides a number of methods that are able to iterate over any Collection. Each Java Collection contains the Iterator method that returns an Iterator instance. Iterators are capable of removing elements from the underlying collection during the iteration.

</details>

<details><summary><b> How HashMap works in Java? </b></summary>

> A HashMap in Java stores key-value pairs. The HashMap requires a hash function and uses hashCode and equals methods, in order to put and retrieve elements to and from the collection respectively. When the put method is invoked, the HashMap calculates the hash value of the key and stores the pair in the appropriate index inside the collection. If the key exists, its value is updated with the new value. Some important characteristics of a HashMap are its capacity, its load factor and the threshold resizing.
  
</details>


<details><summary><b>What differences exist between HashMap and Hashtable? </b></summary>

> Both the HashMap and Hashtable classes implement the Map interface and thus, have very similar characteristics. However, they differ in the following features:

    A HashMap allows the existence of null keys and values, while a Hashtable doesn’t allow neither null keys, nor null values.
    A Hashtable is synchronized, while a HashMap is not. Thus, HashMap is preferred in single-threaded environments, while a Hashtable is suitable for multi-threaded environments.
    A HashMap provides its set of keys and a Java application can iterate over them. Thus, a HashMap is fail-fast. On the other hand, a Hashtable provides an Enumeration of its keys.
    The Hashtable class is considered to be a legacy class.

  
</details>


<details><summary><b>What do you know about the big-O notation and can you give some examples with respect to different data structures? </b></summary>

> The Big-O notation simply describes how well an algorithm scales or performs in the worst case scenario as the number of elements in a data structure increases. The Big-O notation can also be used to describe other behavior such as memory consumption. Since the collection classes are actually data structures, we usually use the Big-O notation to chose the best implementation to use, based on time, memory and performance. Big-O notation can give a good indication about performance for large amounts of data.
  
</details>


<details><summary><b>  What is the purpose of garbage collection in Java, and when is it used?  </b></summary>

> The purpose of garbage collection is to identify and discard those objects that are no longer needed by the application, in order for the resources to be reclaimed and reused.

</details>

<details><summary><b>What does System.gc() and Runtime.gc() methods do?  </b></summary>
 
> These methods can be used as a hint to the JVM, in order to start a garbage collection. However, this it is up to the Java Virtual Machine (JVM) to start the garbage collection immediately or later in time.

</details>


<details><summary><b>When does an Object becomes eligible for Garbage collection in Java ?  </b></summary>

> A Java object is subject to garbage collection when it becomes unreachable to the program in which it is currently used.

</details>


<details><summary><b> What is the difference between Exception and Error in java?  </b></summary>

> Exception and Error classes are both subclasses of the Throwable class. The Exception class is used for exceptional conditions that a user’s program should catch. The Error class defines exceptions that are not excepted to be caught by the user program.

</details>


<details><summary><b>  What is the importance of finally block in exception handling?  </b></summary>

> A finally block will always be executed, whether or not an exception is actually thrown. Even in the case where the catch statement is missing and an exception is thrown, the finally block will still be executed. Last thing to mention is that the finally block is used to release resources like I/O buffers, database connections, etc.

</details>


<details><summary><b> What will happen to the Exception object after exception handling? </b></summary>

> The Exception object will be garbage collected in the next garbage collection.

</details>


<details><summary><b> What is an Java Applet?  </b></summary>

> A Java Applet is program that can be included in a HTML page and be executed in a java enabled client browser. Applets are used for creating dynamic and interactive web applications.

</details>


<details><summary><b> What is a layout manager?</b></summary>

> A layout manager is the used to organize the components in a container.

</details>


<details><summary><b> How can a GUI component handle its own events? </b></summary>

> A GUI component can handle its own events, by implementing the corresponding event-listener interface and adding itself as its own event listener.

</details>

<details><summary><b> What advantage do Java’s layout managers provide over traditional windowing systems? </b></summary>

> Java uses layout managers to lay out components in a consistent manner, across all windowing platforms. Since layout managers aren’t tied to absolute sizing and positioning, they are able to accomodate platform-specific differences among windowing systems

</details>


<details><summary><b>  What is the design pattern that Java uses for all Swing components?  </b></summary>

> The design pattern used by Java for all Swing components is the Model View Controller (MVC) pattern.

</details>


<details><summary><b> What is JDBC? </b></summary>

> JDBC is an abstraction layer that allows users to choose between databases. JDBC enables developers to write database applications in Java, without having to concern themselves with the underlying details of a particular database.

</details>


<details><summary><b>  What is the purpose Class.forName method? </b></summary>

> This method is used to method is used to load the driver that will establish a connection to the database.
  
</details>


<details><summary><b> How are the JSP requests handled?  </b></summary>

> On the arrival of a JSP request, the browser first requests a page with a .jsp extension. Then, the Web server reads the request and using the JSP compiler, the Web server converts the JSP page into a servlet class. Notice that the JSP file is compiled only on the first request of the page, or if the JSP file has changed.The generated servlet class is invoked, in order to handle the browser’s request. Once the execution of the request is over, the servlet sends a response back to the client

</details>


<details><summary><b>  What are Directives? </b></summary>

> What are the different types of Directives available in JSP ? Directives are instructions that are processed by the JSP engine, when the page is compiled to a servlet. Directives are used to set page-level instructions, insert data from external files, and specify custom tag libraries. Directives are defined between < %@ and % >.The different types of directives are shown below:

    Include directive: it is used to include a file and merges the content of the file with the current page.
    Page directive: it is used to define specific attributes in the JSP page, like error page and buffer.
    Taglib: it is used to declare a custom tag library which is used in the page.


  
</details>


<details><summary><b> What are JSP actions?  </b></summary>
 
> JSP actions use constructs in XML syntax to control the behavior of the servlet engine. JSP actions are executed when a JSP page is requested. They can be dynamically inserted into a file, re-use JavaBeans components, forward the user to another page, or generate HTML for the Java plugin.Some of the available actions are listed below:

    jsp:include – includes a file, when the JSP page is requested.
    jsp:useBean – finds or instantiates a JavaBean.
    jsp:setProperty – sets the property of a JavaBean.
    jsp:getProperty – gets the property of a JavaBean.
    jsp:forward – forwards the requester to a new page.
    jsp:plugin – generates browser-specific code.


  
</details>


<details><summary><b> What are Decalarations? </b></summary>

> Declarations are similar to variable declarations in Java. Declarations are used to declare variables for subsequent use in expressions or scriptlets. To add a declaration, you must use the sequences to enclose your declarations.

</details>


<details><summary><b>What are Expressions?  </b></summary>
  
> A JSP expression is used to insert the value of a scripting language expression, converted into a string, into the data stream returned to the client, by the web server. Expressions are defined between <% = and %> tags.

</details>


<details><summary><b> Explain the architechure of a Servlet </b></summary>

> The core abstraction that must be implemented by all servlets is the javax.servlet.Servlet interface. Each servlet must implement it either directly or indirectly, either by extending javax.servlet.GenericServlet or javax.servlet.http.HTTPServlet. Finally, each servlet is able to serve multiple requests in parallel using multithreading.

</details>


<details><summary><b>What is meant by a Web Application?  </b></summary>

> A Web application is a dynamic extension of a Web or application server. There are two types of web applications: presentation-oriented and service-oriented. A presentation-oriented Web application generates interactive web pages, which contain various types of markup language and dynamic content in response to requests. On the other hand, a service-oriented web application implements the endpoint of a web service. In general, a Web application can be seen as a collection of servlets installed under a specific subset of the server’s URL namespace.

</details>


<details><summary><b>  What’s the difference between sendRedirect and forward methods? </b></summary>

> The sendRedirect method creates a new request, while the forward method just forwards a request to a new target. The previous request scope objects are not available after a redirect, because it results in a new request. On the other hand, the previous request scope objects are available after forwarding. FInally, in general, the sendRedirect method is considered to be slower compare to the forward method.

</details>


<details><summary><b> Explain Serialization and Deserialization </b></summary>
 
> Java provides a mechanism, called object serialization where an object can be represented as a sequence of bytes and includes the object’s data, as well as information about the object’s type, and the types of data stored in the object. Thus, serialization can be seen as a way of flattening objects, in order to be stored on disk, and later, read back and reconstituted. Deserialisation is the reverse process of converting an object from its flattened state to a live object.

</details>


<details><summary><b> Explain breadth first and depth first search </b></summary>

> BFS is a traversing algorithm where you should start traversing from a selected node (source or starting node) and traverse the graph layerwise thus exploring the neighbour nodes (nodes which are directly connected to source node). You must then move towards the next-level neighbour nodes.

> The DFS algorithm is a recursive algorithm that uses the idea of backtracking. It involves exhaustive searches of all the nodes by going ahead, if possible, else by backtracking.

</details>

<details><summary><b> Explain Linear Search </b></summary>

> Linear search is used on a collections of items. It relies on the technique of traversing a list from start to end by exploring properties of all the elements that are found on the way. 

</details>

<details><summary><b> Explain binary search </b></summary>

> Search a sorted array by repeatedly dividing the search interval in half. Begin with an interval covering the whole array. If the value of the search key is less than the item in the middle of the interval, narrow the interval to the lower half. Otherwise narrow it to the upper half. Repeatedly check until the value is found or the interval is empty.

</details>

<details><summary><b> Explain Ternary search </b></summary>

> is a searching technique that is used to determine the position of a specific value in an array. In binary search, the sorted array is divided into two parts while in ternary search, it is divided into 3 parts and then you determine in which part the element exists.  

  
</details>


<details><summary><b>Explain bubble sort </b></summary>

> Bubble sort is based on the idea of repeatedly comparing pairs of adjacent elements and then swapping their positions if they exist in the wrong order. 
> [ans](https://www.hackerearth.com/practice/algorithms/sorting/bubble-sort/tutorial/)
  
</details>


<details><summary><b> Explain selection sort </b></summary>
 
> The Selection sort algorithm is based on the idea of finding the minimum or maximum element in an unsorted array and then putting it in its correct position in a sorted array.

Assume that the array  A = [7,5,4,2]needs to be sorted in ascending order.

The minimum element in the array i.e. 2 is searched for and then swapped with the element that is currently located at the first position, i.e. 7. Now the minimum element in the remaining unsorted array is searched for and put in the second position, and so on.

> [more](https://www.hackerearth.com/practice/algorithms/sorting/selection-sort/tutorial/) 

</details>


<details><summary><b> Explain insertion sort </b></summary>

> Insertion sort is based on the idea that one element from the input elements is consumed in each iteration to find its correct position i.e, the position to which it belongs in a sorted array.

It iterates the input elements by growing the sorted array at each iteration. It compares the current element with the largest value in the sorted array. If the current element is greater, then it leaves the element in its place and moves on to the next element else it finds its correct position in the sorted array and moves it to that position. This is done by shifting all the elements, which are larger than the current element, in the sorted array to one position ahead

</details>


<details><summary><b> Explain Merge  </b></summary>


> Merge sort is a divide-and-conquer algorithm based on the idea of breaking down a list into several sub-lists until each sublist consists of a single element and merging those sublists in a manner that results into a sorted list. 

</details>


<details><summary><b> What is big 0 notation </b></summary>

> Big O notation is used in Computer Science to describe the performance or complexity of an algorithm. 

</details>



<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>



<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>



<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>


<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>

<details><summary><b> </b></summary>
  
</details>




