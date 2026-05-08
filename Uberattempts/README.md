# Ubermodel
I propose a new framework to train AIs named Ubermodel. Its name as well as processes derieve from Nietzsche's Ubermensch, Netiflix's Squid Games and Darwin's Survival of the Fittest. 

## V1
This is the first formulation of the Ubermodel. But to fix some conceptual things, I would need to revise some things too.
### Generating population
Generate $n$ models in a set $\Phi$. Which makes it obvious that $|\Phi| = n$ and
$$
\Phi = \{\Psi_1, \Psi_2, \ldots\}
$$
Or 
$$
\Phi = \{\Psi_i: \forall i \in \mathbb{N}^+ \leq n\}
$$

### Models
Let each model $\Psi$ have the following parameters:
- $\nu$ - parameters
    * These are the normal parameters a model usually has. I am grouping all of these under the umbrella term $\nu$. These would be things like learning rate, or other things.
- $f$ -- Defined in $\S{f}$ 

Therefore each model is defined as
$$
\Psi_i = (\nu_i, f_i)
$$
// **todo: look for a better symbol for $f$. Feoh sounds like a good alternative, but John McArthy would hire a hitman on me if I did that.**

### $f$
$f$ was used in the definition of a model. It is definied as
$$
f_i = g(\Psi_i)
$$
but that can still cause some confusion. Since $f$ is a part of the model's definition and calculated on it. So maybe either I do $\Psi_i = \Psi(\nu_i)$ where $\Psi$ is a "model-creator" of sorts (ie. itd be something along the lines of generating the $\nu$.) or I could do $f = g(\nu)$. I think, the second one would be better. But they are essentially doing the same thign. Therefore
$$
f_i = g(\nu_i)
$$
where $g$ is arbitrary (See $\S{g}$).

### $g$
I said before that $g$ is arbitrary. But that was not accurate to what I was going for. A better would have been that **$g$ is a function defined by the user (programmer, in this case) that does something to $\nu$**. 

I advice that for serious applications different $g$'s should be used. So, for instance say there is
- $g_1 = \sum\nu$
- $g_2 = \Pi\nu$
- $g_3 = \bar{\nu}$
etc.

These are very simple examples, but for serious use cases these could be as complex as one can imagine. 

I recommend that once one $g$-function is used, then use another on the same starting models. Put simply, change $g$ after a complete run.

Side note: Since $\nu$ is an umbrella term, the actual parameters (say, learning rate) within $\nu$ would be operated upon as follows
$$
g = \{\sum\nu_1, \sum\nu_2, \ldots\}
$$
where $\nu_1, \nu_2, \ldots$ are the different parameters within $\nu$. 

Essentially
$$
g: \nu \rightarrow \mathbb{R}
$$

### Grouping
Let $\lambda$ be the grouping operator as:
$$
\lambda(\psi_i, \psi_j)=
\begin{cases}
|f_i-f_j| < k, & 1 \\
\text{otherwise}, & 0
\end{cases}
$$

So essentially, 1 and 0 I've used for true and false here. Since they would make more sense mathematically and later it'd be easier in local comparisions. $k$ is yet undefined. It could be either a user-defined constant (preferably very small, judging from my experiments) or I might provide a definition for it. For now just assume $k \approx 0$. The rest is pretty self explanatory. 

If they satisfy the conditions, the models will be in the same group $\epsilon$. I could use either $\epsilon_1, \ldots$ or $\epsilon_{i,j}$. I am still quite unsure of it. For now I'll use $\psi_i, \psi_j \in \epsilon_1$. so an epsilon would be defined formally as
$$
\epsilon_r = \{\Psi_i\in\Phi: \lambda(\Psi_i, \Psi_j) = 1\}
$$

### Comparision
Let $\chi$ be the comparision controller. Important is that it is an umbrella term too. Under it would be comparision of many different metrics like accuracy, speed, efficiency, etc. 

To define the best model, we would need a metric for "how good" the model is. Let's call it...$\tau$. It is quite different from $f$ since $f$ "decides who **competes**" and $\tau$ "decides who **wins**".

$$
\tau_i = \alpha({A}) - \beta({B}) - \gamma({C}) \ldots
$$
where $A, B, C \ldots$ are for example Asagrim, Bilyegr and Svafnir. These are just examples to show that pretty much anyhting, like accuracy, latency, cost etc. And $\alpha, \beta, \gamma \ldots$ are importance controllers (so "how much" to prioritise the corresponding metrics matter). 

note that you could also do $+$ instead of $-$, $\times$ too if you want, any operand.

* $\tau_i$ is one notation, other can be $\tau(\Psi_i)$ which is what $\tau_i$ represents, just that the latter is shorteneed for convinience. 

So now we can compare the models under $\chi$ as
$$
\chi(\epsilon_r) = \argmax_{\Psi_i\in\epsilon_r}\tau(\Psi_i) = \Psi_w
$$
where, self-explanatorily, $\Psi_w$ is the winner model.

### Redistribution
Let $\rho$ be the redistribution operator. All it does is
$$
\forall \Psi_i \in \epsilon_r \setminus \{\Psi_w\}: 
\begin{aligned}
f'_i &= f_i + \eta(f_w - f_i) + \eth_f \\
\nu'_i &= \nu_i + \eta(\nu_w - \nu_i) + \eth_\nu
\end{aligned}
$$
where $\eta$ is a redistribution controller, so essentially "how much" to redistribute. And $\eth_f, \eth_\nu$ are to add some noise to preserve diversity and avoiding too fast convergence. 

The rest of the operation seems pretty self explanatory to me, it would be done using the `+=` operator in code. So if I go a bit out of my lane it would be 
```python
f_i += eta(f_w - f_i) + eth_f
nu_i += eta(nu_w - nu_i) + eth_nu
```
I used two $\eth$'s to add noise to both parameters, but it could be achieved with just one as well, just if you want more flexibility and contol, two $\eth$'s would be a solution.

This redistribution is one of teh best things about this algorithm, it preserves divesity. 

### Some addons
To be clear, these are not just random extra features, these are part of the core working. These may include redefinitons or clarifications.
- We can make the size of each $\epsilon$ bigger. 
    * For now there are only two $\Psi \text{ per }\epsilon$. We can make it bigger. It might cause a problem that all models just get grouped in one, but to avoid it we can do $2\leq|\epsilon|\leq{m}$ where $m$ is the maximum group size. And to group models we can do a dynamic $k$. Such that $k_i = \text{distance to the m-th nearest neighbour}$. THis makaes it that denser regions get tighter neighbourhoods and the converse. It implicitely preserves exploration too.

- Deletion
    * This is a problem because the model describe till now just does stuff, it would be endless. So we need deletion. I can do two things. Either delete the worst models or keep only the best models. I think both would make a good bet. But in a logical way. Let $t$ be the iteration index so the number of iteration you are currently on. Then
$$
|\Phi_{t+1}| = a_t|\Phi_t| \text{ where } 0 < a_t < 1 \text{ and } a_t \rightarrow 1 \text{ over time}
$$

> which would mean shrink the population. $a_t$ is less than 1 so it can't grow the population. And when it equals 1, the population later becomes equal to population now ie. $|\Phi| = m$. To be clear, we delete the worst models every iteration. Local worst models. 

### Conclusion
Continue this until one model emerges perfect. Since when $|\Phi| = m$, $\chi$ would be run once more, and the best model would be $\Psi_w$. Then, if you want, run it again with a differnt choice of $g$. After completeing that too, you can compare the models produced by different $g$'s.

## V2
not necesserily improvements, but extra features for some edge cases or more robust applications.
### Neighbourhoods/Grouping
Now I need to make grouping temporary. I originally intended it to be this way, but I thought I should explicitely mention it. Therefore
$$
\epsilon^{t+1} \neq \epsilon^t
$$
This was originally what i was going for. This avoids static "tribe" like groups and helps in exploration.

### Multidimensional $f$
So right now $f$ is a single, scalar, boring real number. But that is bleh. In academic terms "restrictive". So I think $f$ should be multidimensional
$$
f_i \in \mathbb{R}^d
$$
where $d$ is the number of dimensions. So this can actually encode a lot more of $\nu$ since $\nu$, too, isnt' a single quantity. 

### Anti-singularity
Right now, the algorithm naturally tends towards all models becoming identical (or..similar). But, while that may be useful in some niche cases, I think I should have a general approach too. I shouldn't have it go towards one-ness, i want something that can reward novelty as well as good models.

Therfore I introduce a new operator for diversity.
$$
\tau'_i = \tau_i + \lambda{D_i}
$$
where $D_i$ is the divesity score(r) and $\lambda$ is a controller ("how much" diversity).

Let $D_i$ be defined as something *simple* like
$$
D_i = \frac1{|\epsilon_r|}\sum_{\Psi_j\in\epsilon_r}|f_i - f_j|
$$
essentially, mean distance from neighbours.

Now important note, this is not entirely necessery. As I said, some niche/edge cases can mangae without this. But, if ever there is too fast covergence (like those in `..\Failed attempts that might be useful`") then this might be useful. 

### Memory
Right now the models only "remember"$^*$ the current neighbourhoods. But what if we add memoery such that the models that win repeatedly become more "influential", the models that lose repeatedly get penalised. 
$$
\tau'_i = \tau_i + \mu\zeta_i
$$
where $\zeta_i$ is the historical performance and $\mu$ is obviusly again "how much". 

note that $\zeta_i$ should be $\zeta_i^{(t)}$ sicne every iteration it would change the memory. 

I should define $\zeta_i^t$ too. So it is:
$$
\zeta_i^{(t+1)} = (1 - \omega)\zeta_i^{(t)} + \omega\tau_i^{(t)}
$$
where if $\omega\rightarrow{1}$, then recent performance matters, memory updates faster because the $(1 - \omega)\zeta_i^{(t)}$ value becomes less since $(1-(\approx{1}))\zeta_i^{(t)} \implies \zeta_i^{(t)}\rightarrow{0}$. 

Conversely, if $\omega\rightarrow{0}$, then long term performance matters since $(1-(\approx 0))\zeta_i^{(t)} \implies \zeta_i^{(t)} \rightarrow{1}$.

This is good because without this it can be that just one "lucky" model wins, but with this consistent performace would matter. 

Also $\mu$ should be relatively moderate. Because it can make new models and exploration difficult to emerge and have a sort of tyranny of old models.

### Overlapping neighbourhoods
One limitation of the current neighbourhood system is that neighbourhoods are discrete. A model either belongs to a neighbourhood or it does not. This may create rigid boundaries in the optimization space, causing abrupt changes in interactions and reducing smooth exploration of nearby regions. It may also cause unstable grouping when many models have similar $f$-values near the grouping threshold $k$. Therefore
$$
w_{i,j} = \exp\big(-\frac{|f_i - f_j|^2}{k}\big)
$$
where $w_{i,j} \in [0,1]$. Models with closer $f$ have stronger interaction weights, and vice versa.

btw $\exp$ means power of $e$ so it can be still written as $e^{\big(-\frac{|f_i - f_j|^2}{k}\big)}$ but it gets clunky so I use $\exp$.

This changes the neighbourhood system from rigid groups into overlapping local ecosystems. A model may strongly interact with one neighbourhood while weakly interacting with another. Such overlapping neighbourhoods preserve smoother exploration of the optimization space and reduce sudden convergence effects. So we can write redistribution as
$$
f'_i = f_i + \eta\sum_j w_{i,j}(f_j - f_i) + \eth_f
$$
where nearby models contribute more strongly to redistribution than distant ones.

Similarly redistribution for $\nu$ becomes
$$
\nu'_i = R(\nu_i, \sum_j w_{i,j}\nu_j, \eta, \eth_\nu)
$$
where $R$ is the redistribution operator. 

This formulation transforms Ubermodel from a discrete neighbourhood optimizer into a topology-aware adaptive interaction system, where optimization emerges through weighted local influence rather than strict grouping boundaries.

### Clarification of etymology
I said before this derives from Darwin's Survival of the Fittest, but now the framework, logically so, tends more towards survival through adaptive local interaction. It is quite too academic, but simply it means weak models improve through redistribution, neighbourhoods evolve and winners influence everything. 

### Termination
So how should it end. IT should end, not when there is no change. Because mathematically there always can be change. We want that when there is no meaningful change, we don't stop, we, academically, ***terminate***. So
$$
\text{Terminate: } \\
\text{If } \forall\Psi_i\in\Phi: |f_i^{t+1} - f_i^{(t)}| < \eth_f \\ 
\text{And } |\tau_\text{max}^{(t+1)} - \tau_\text{max}^{(t)}| < \eth_\tau \\
$$
where $\eth_\tau$ is a small number, actual mathematics would have used $\epsilon$ but I have used that symbol elsewhere and used $\eth$ for small noise number for the thing.