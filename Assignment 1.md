
ASSIGNMENT 1
Consider the following schema:
SUPPLIERS (SID: integer, SNAME: string, STREET: string, CITY: string, ZIP: string) PARTS (PID: integer, PNAME: string, COLOR: string)
CATALOG (SID: integer, PID: integer, COST: real)
The primary key attributes are underlined, and the domain of each attribute is listed after the attribute name. Thus, SID is the primary key for SUPPLIERS, PID is the primary key for PARTS, and SID and PID together form the primary key for CATALOG. Attribute SID in CATALOG is a foreign key that refers to the primary key SID of SUPPLIERS and PID in CATALOG is a foreign key that refers to the primary key PID of PARTS. The CATALOG relation lists the prices charged for parts by suppliers.

State what the following query computes.
0.	π	(π	(σ	(PARTS) ) * σ	(CATALOG) * π	(SUPPLIERS) )
CITY	PID	COLOR = 'red'	COST>100	SID, CITY

Write the following queries in relational algebra.
1.	Find the names of parts for which there is some supplier
2.	Find the names of parts supplied by suppliers who are at 1 Central Ave.
3.	Find the names of suppliers who supply some red part.
4.	Find the SIDs of suppliers who supply some red or green part.
5.	Find the SID of suppliers who supply some red part or whose address is '221 Packer Street'.
6.	Find the SIDs of suppliers who supply some red part and some green part.
7.	Find the PIDs of parts that are red or are supplied by a supplier who is at the city of Newark.
8.	Find the PIDs of parts supplied by a supplier who is at the city of Newark and by a supplier who is at the city of Trenton.
9.	Find the PIDs of parts supplied by every supplier.
10.	Find the PIDs of parts supplied by every supplier who supplies at least one part.
11.	Find the PIDs of parts supplied by every supplier who is at the city of Newark or at the city of Trenton (equivalently: find the PIDs of parts supplied by every supplier who is at the city of Newark and by every supplier who is at the city of Trenton).
12.	Find the PIDs of parts supplied by every supplier who is at the city of Newark or by every supplier who is at the city of Trenton.
13.	Which one of the queries 11 and 12 is more restrictive (if any)?
14.	Find pairs of PIDs such that the part with the first PID is sold at a higher price by a supplier than the part with the second PID.
15.	Find the SIDs of suppliers who supply at least two different parts (you are not allowed to use a grouping/aggregation operation for this query).
16.	Find the SIDs of suppliers who supply at least two different parts (you have to use a grouping/aggregation operation for this query).
17.	For every part supplied by a supplier who is at the city of Newark, print the PID and the SID and the name of the suppliers who sell it at the highest price.
18.	For every part, find its PID, its PNAME and the number of suppliers who sell it.
19.	List the PID, PNAME and average cost of all parts.
20.	Find the average cost of red parts.
21.	Find the average cost of parts supplied by suppliers named 'Yosemite Sham'.
