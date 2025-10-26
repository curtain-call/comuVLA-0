### Assignment 1 — 关系代数解答（仅答案）

0)
存在供应商以价格>100销售某红色零件的城市集合

1)
```text
π_PNAME( PARTS ⋈ CATALOG )
```

2)
```text
π_PNAME( σ_STREET='1 Central Ave.'(SUPPLIERS) ⋈ CATALOG ⋈ PARTS )
```

3)
```text
π_SNAME( SUPPLIERS ⋈ CATALOG ⋈ σ_COLOR='red'(PARTS) )
```

4)
```text
π_SID( CATALOG ⋈ σ_COLOR∈{'red','green'}(PARTS) )
```
```text
π_SID( CATALOG ⋈ σ_COLOR='red'(PARTS) ) ∪ π_SID( CATALOG ⋈ σ_COLOR='green'(PARTS) )
```

5)
```text
π_SID( CATALOG ⋈ σ_COLOR='red'(PARTS) ) ∪ π_SID( σ_STREET='221 Packer Street'(SUPPLIERS) )
```

6)
```text
π_SID( CATALOG ⋈ σ_COLOR='red'(PARTS) ) ∩ π_SID( CATALOG ⋈ σ_COLOR='green'(PARTS) )
```

7)
```text
π_PID( σ_COLOR='red'(PARTS) ) ∪ π_PID( σ_CITY='Newark'(SUPPLIERS) ⋈ CATALOG )
```

8)
```text
( π_PID( σ_CITY='Newark'(SUPPLIERS) ⋈ CATALOG ) ) ∩ ( π_PID( σ_CITY='Trenton'(SUPPLIERS) ⋈ CATALOG ) )
```

9)
```text
π_PID( π_SID,PID(CATALOG) ÷ π_SID(SUPPLIERS) )
```

10)
```text
π_PID( π_SID,PID(CATALOG) ÷ π_SID(CATALOG) )
```

11)
```text
π_PID( π_SID,PID(CATALOG) ÷ π_SID(σ_CITY='Newark'(SUPPLIERS)) ) ∩ π_PID( π_SID,PID(CATALOG) ÷ π_SID(σ_CITY='Trenton'(SUPPLIERS)) )
```

12)
```text
π_PID( π_SID,PID(CATALOG) ÷ π_SID(σ_CITY='Newark'(SUPPLIERS)) ) ∪ π_PID( π_SID,PID(CATALOG) ÷ π_SID(σ_CITY='Trenton'(SUPPLIERS)) )
```

13)
```text
Query 11 更严格
```

14)
```text
π_{C1.PID, C2.PID}( σ_{C1.SID=C2.SID ∧ C1.COST > C2.COST}( ρ_C1(CATALOG) × ρ_C2(CATALOG) ) )
```

15)
```text
π_{C1.SID}( σ_{C1.SID=C2.SID ∧ C1.PID ≠ C2.PID}( ρ_C1(CATALOG) × ρ_C2(CATALOG) ) )
```

16)
```text
π_SID( σ_cnt≥2( γ_{SID; COUNT(PID)→cnt}( CATALOG ) ) )
```

17)
```text
π_{PID, SID, SNAME}( ( σ_{COST=maxC}( CATALOG ⋈ γ_{PID; MAX(COST)→maxC}( CATALOG ) ) ⋈ SUPPLIERS ) ⋈ π_PID( σ_CITY='Newark'(SUPPLIERS) ⋈ CATALOG ) )
```

18)
```text
γ_{PID, PNAME; COUNT(SID)→numSuppliers}( CATALOG ⋈ PARTS )
```

19)
```text
γ_{PID, PNAME; AVG(COST)→avgCost}( CATALOG ⋈ PARTS )
```

20)
```text
γ_{ ; AVG(COST)→avgRedCost}( CATALOG ⋈ σ_COLOR='red'(PARTS) )
```

21)
```text
γ_{ ; AVG(COST)→avgCost}( CATALOG ⋈ σ_SNAME='Yosemite Sham'(SUPPLIERS) )
```


