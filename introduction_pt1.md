# Introduction Part 1

## PNRGs

* for uniform dists
* difficult to distinguish

### Linear congruential generators:

* construct uniformly distributed random numbers on o, m-1
* so seed deterministic transformation $f:\{0,\ldots,m-1\}\rightarrow\[0,\ldots,m-1\}$
* $\Rightarrow$ seq. of PRN $s_n=f(s_{n-1})$
* in particular: $s_n=a\cdot s_{n-1}+c \mod m$
* $\Rightarrow$ divide by $m-1$ to generate random numbers which are independent and approx uniformly distributed in $\\[0,1\\]$
* performance crucially depends on chosen parameters $a, c \& m$
* $m$ possible states of the random numbers 
* $\Rightarrow$ periodic seq. after $m$ steps or earlier
* $\Rightarrow$ new seed needed

* QQ: Finite Fields? Cryptography

### General methods for generation of random numbers

1. Inverse transformation (Wt0 Dahlhaus Ex.19, Sheet 05)
Def: CDF F 

Prp: Properties of F

Def: Let $F$ be a cdf. The __Pseudoinverse__ of $F$ is defined by 
$$I_F:\\[0,1\\]\to\mathbb{R}$$ with 
$$I_F(y):=\inf\{x\in\mathbb{R},F(x)\geq y\}$$


#### Properties:
1. $I_F$ is 
2. $I_F(y) \leq x \iff y\leq F(x) (\forall x\in\mathbb{R},y\in\\[0,1\]])$
3. $F(x)=\sup\{y\in\\[0,1\\],I_F(y)\leq x\},x\in\mathbb{R}$
4. $F(I_F(y))\geq y$
5. $I_F(F(x))\leq x$

Prp: Let X be RV with cdf $F_X$ and let $U\sim U(\\[0,1\\])$
Then $I_{F_X}\circ U$ has $F_X$ as its CDF and therefore same __law__ (dist) as $X$.

#### Algorithm
1. generate $u\simU([0,1])$
2. set $x=I_F(u)$

Example: see (Wt0 Dahlhaus Ex.19, Sheet 05)







