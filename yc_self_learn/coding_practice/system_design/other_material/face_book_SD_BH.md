Algo:

Leetcode problems list:
https://www.one-tab.com/page/Wa0dus32Rv6iyMoHArD-QQ

以下小部分不在leetcode，大部分其实在但是我懒得找了：

https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=318820&highlight=battleship
str to datetime
Remove minimal parentheses to make it valid 删除最少括号
Find longest path in tree(parent->children and children->parent)
给一个字符串，是一串合法括号。把这个转换成多叉树。每一个被嵌套的括号是外面一层括号的子节点。
不难，我一开始本能想写dfs，慌慌张张写了半天实现不出来。但是面试官非常好，一直在follow我的思路。然后又给了提示，直接扫一遍，树节点里带上母节点指针就好了
 find the largest permutation less than input
利口42（盛水题）。给一个数组，每个代表高度，问最大能存多少水。时间、空间开销
第二题：假如小于等于0表示黑洞，水都会流走，问最大能存多少水。时间、空间开销
给一个数组，打印所有的组合。时间、空间开销
一个array里面都是正数，找是否存在一个sub array里面的数的和等于target，返回true、false
一个matrix里面只有1或0。实现一个function， 给x1,y1, x2,y2, x1y1代表矩阵左上角坐标，x2y2代表矩阵右下角坐标，求从x1y1到x2y2的矩阵里面的1的个数。要求是这个矩阵里面的值不变，这个function可能会被call很多很多次。
第一问：给一个string，和一个dictionary，求string能否由dictionary中的两个word组成。返回true/false。follow up是如果这个string可能由多个word组成，怎么做。我就口头上说了一下用trie存dictionary，就去下一道题了
 一个interval表示一条红色胶带，一个list of interval表示一条条黑色胶带。假设黑色胶带和红色胶带重叠的话可以完全遮挡住红色胶带，求最后能否看见红色胶带，返回true/false。黑色胶带的interval需要自己处理一下，不是都是merge好的最终形态。
给一棵n叉树，要求把这棵树分割成多个subtree，使每个subtree里面的node数都是偶数个。要求把一棵树尽可能的多分。输出自己定义
 把二叉树添加next节点，指向右侧节点，follow up：不用额外空间
bipartite graph
给一个都是正数的数组，求size为k的最大和。求三个size为k的最大和
利口674，follow up是最多可以删除一个数时，求最长
https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=535731
 二叉树打印从parent 到所有leaf nodes路径。 dfs. 注意避免重复打印同一路径. path是list。 用于backtracking
give a array of points, [[x,y], [ ...., find all four points that can formulate a rectangular ( assume all edges are parallel to x and y)    extension, what if edges not parallel to x and y ( hint: find slope k and -1/k edges)
interval的题是说array of interval，比如【【1，3】，【2，3】，【2，5】，【6，8】】每个interval都是闭合的，然后return被cover最多次的那个integer（有多个就任意返回一个）。这题的解法就是把所有的区间的start和end + 1算event，start + plus或者（end + 1）+ sub。sort了所有的event然后从头到尾对于counter（初始为0）要add就add，要sub就sub，记录最大的counter和对应的那个start value（必然有一个start value符合条件）。当然你做些处理也可以算出interval的
check if a string list/array is in a custom order: arr = ['abc', 'bc', 'cd'..] , order = ['c', 'b', 'a']
Interleave two linked lists
The right view of binary tree
find distance between any two nodes in a binary tree
distance between any two nodes， 就是把tree看成graph，找到两个node间距离。No parent pointer, need less than O(n) space.
https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=531098
Kth largest Number in an unsorted array
bst->doubly linked list。有点蒙，我最后用的是tree iterator（stack）做的。花了一些时间解释为啥我那个stack可以起到＋＋效果，他表示理解了。我最后写完了，中间自己debug了几个小地方。
利口15（3Sum）只需要true/false
取出所有array里local minimum的index，边界也要考虑
无限大棋盘，从a走到B。棋盘大，所以不能用2d array来表示。中间有障碍。问a是否可以走到B。有两个 api可以问比如canMove（）, isB（)
卡住了。先上bfs，不行，greedy不行。最后写了skeleton 的dfs。他觉得dfs是可行的。其实。。。俺最后还是有点不明觉厉。。这轮要是跪了估计没啥好说的。。。
https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=531037
打印2d矩阵的斜对角
一个array很长，只包含1和0， 比如[1,0,0,1,1,0,0,0..........1] 要query得到range[left,right]之间有多少个1， 要求query要很快，我说hashmap存一下index和这个位置往前一共有多少个
有一个maze，里面有一个老鼠M，有一个或者没有cheese，设计一个method返回是否M可以找到这个cheese
利口53（Maximum Subarray）
给定两个editor 的input 操作（输入char或者delete '\b'）判断最后output 是否相等
topological tree, given courses and their dependencies, whether you can take all courses, return boolean
从preorder list build 得到 inorder的顺序
给你一个数组int[] input， 数组input是通过调用N次（N是数组长度）辅助函数int f(int[] sorted)在一个有序数组sorted形成的，辅助函数每次生成有序数组的最大值或者最小值，每次调用辅助函数会remove这个element。要求实现算法 找到有序数组sorted。
利口621变种，但是要按照给予的task的order顺序执行task。e.g. [A,B,A,B] cool down = 2. output must be [A, B, _, A, B]. return length = 5.
类似利口121，假设要出发去某地，两个input array分别是去程票价和回程票价。求最小的花费，可以同一天出发同一天返回
e.g. dept prices i on day i [10, 20, 15]; return price i on day i [21, 7, 10], 那么‍‌‌‌‌‌‌‌‌‍‍‍‍‍‌‍‌ 最小花费是17，我们day0 出发, day1 返回.
利口15（3Sum），不过给的是数组，follow up如何提速，我只答出个碰到0就越过这次循环，在她的提示下，答出可以把数组拆解成几个小数字，然后数字之间相乘。
利口1（2Sum）变形，给一个2d grid，每个格子的value，1是障碍，w指仓库，0是空格，找一个空格使得所有仓库离他的距离最小
类似利口986，不过比那个要难很多。给一个list of intervals，找interval重叠最多的interval，应该是用Sweep Line做。这题没见过，一时思路没跟上来，解释了很长时间我的想法，韩裔小哥一直在帮助我，我感受到了。不过最后还是对Sweep Line不熟悉，没想出最好解法。说了一个暴力解法
Coding2: LC350，用HashMap做了。Follow-up1: 有没别的做法，解释了双指针。Follow-up2：比较两种做法时间复杂度优劣，数据量达到什么tipping point的时候一种方法比另一种方法快（给出数学等式）。Follow-up3: 假设两个array中的元素有大量的连续整数的话如何优化，比如a = [(1 5), (8, 10), (15 20)]，b = [(2 6), (45 100)]：return [(2, 5)]（这个例子从另一个面经上找的），没什么思路。
LC560，区别是只需要打印true/false。Follow-up1: 有0怎么办。Follow-up2：有负数怎么办。
不知道有没有LC原题。一个二叉树，给出两个node和最近公共祖先之间（两条path）的sum。写了两个方法， 方法A找最近公共祖先，方法B算二叉树两个node之间的和。先跑方法A，再跑两次方法B。
这轮应该是用来train新的面试官（年轻小印妹子+美国大叔shadow），结果不作为给不给offer的依据。楼主面了一天脑子晕得厉害，有些follow-up我没思路硬说原本的方法就可以work，小印给反例想证明我是错的，没成功，慌张说好吧那下一题。
sparse vector dot product
给你一个string list [cc, cb, bc, ab]和一个custom的字母顺序，比如[c, b, a]，让你返回布尔判断第一个list单词是否严格按照自定义顺序递增。比如例子返回true，因为cc < cb < bc < ab（按第一个不同字符判断）
 [word1, word2, word3...wordn]，就是一串words，这里简写了。找到所有不同的prefix words，words必须是input list里面有的。
利口621（Task Scheduler）类似，但是要求task执行order不能变。
利口674
利口301，只用输出一个结果。
我的解法是扫一遍，用两个stack，一个记录多余的左括号，一个记录多余的右括号。都记录了index。然后把两个stack长度加起来。
follow up: 输出处理后的string. 把stack中存的index跳过就好了。
给你一个数组，k为subarray长度
利口253
利口323，克隆图。要求自定义数据结构，follow up分别是：用最少的空间完成克隆；图有可能有多个连通分支；以及更改input的type，比如给一些node自己判断哪些在一个连通分支里，最后返回一个集合包含复制好的每个连通分支中一个root
写string comparator, 要求是 字母比数字优先级低，如果都是数字就比数字，leading zero不算
fixed order的task schedule， 没写过，写出了一个bug，三哥说了个test case自己改了过来。
task scheduler和利口621的区别是要求按照task array的顺序排序，只准备了的LC上的找max frequency的最优解没法用，现场用hashMap做了一波，面试官说ok。
palindrome at most 1 char
serialize and deserialize binary tree
max diameter of a binary tree, 我用的recursion，然后王者说不喜欢global variable，秒改return pair<int, int>,王者跟我思路不太一样，我又重写了一份neat 版本的。。。被王者夸了句move fast，王者这才很满意拍了照。。
还剩3分钟，拍了王者的马屁假装很感兴趣他的project，然而我已经忘了他是干啥的。。。愉快say goodbye，希望王者给个strong hire
Find number of occurrences of a number 
Find overlap of the two list of intervals 
maximum length of a path of a binary tree
给一个input string和target length返回一个valid html。举个例子，给一个String比如<b>Hello</b>，target length是8，那么返回<b>H</b>。如果target length是9，返回长度为9的valid html，即<b>He</b>。我用的stack记录tag，最后写了一个估计还是有bug的版本。
给一个array of interval，然后一个interval，在这个array里面找到能够cover given interval的最小size 
给一个只含有distinct char的string，打印出所有可能的string，每个string都包含unique char，且是原string char的子集。相同字符集的str只用打印一个。例子：{'abc'} -> {' ', 'a', 'b', 'c', 'ab', 'ac', 'bc', 'abc'}
第二轮：利口42。经典hard题。我是用stack做的，面试官可能没见过这么做的，光解释清楚convince他我是对了就花了十五分钟。讲完之后他还要求我优化，虽然我知道two pointer可以将Space优化到O(1)，无奈没仔细看解法，虽然面试官给了hint也没能急中生智。最后面试官说那你就写stack吧。（哭）写完之后给了follow up：如果这个柱状图里面有个洞（所有这个index的水都会漏掉），要怎么解。
利口415变体，每个string可能是负数。有了负数之后corner case增加了，我一下子没反应过来，没写好。最后没写完就结束了。
coding是给一个年收入和算税金的policy 算出这个人一年交的税 example：给一个 收入 超过 10000 收1%，超过 20000 收2% 。。。。。超过 50000收 5% 再高了就不收了
一道binary search 的题 类似 first bad version
求哪一些边可以组成直角三角形（我一开始用的hashmap，复杂度是O(n²)。面试官问我还能不能优化，还没等我想一会儿，他就说可以用3sum的思路，我立刻知道了，就把解法写给他了）。 
给一段字符串，可以忽略掉一个错误的字母后，是否为回文字符串
利口278（first bad version）
利口98（valid binary search tree）
加强版计算器，先说简单点只有+和*，然后做了，然后说如果+，-，*都有，各种情况都算作valid的输入
利口349/350（intersection of two sorted arrays）。follow up: if one of the arrays is extremely large.
树的序列化和反序列化
给定一个二叉树，随机返回一个节点，要求每个节点返回的概率是一样的。
带timestamp的缓存 (https://leetcode.com/problems/time-based-key-value-store/)
利口25（reverse nodes in k-group）
利口200（数岛），话说这个数岛我今年找工被问了三四次
求一个array里最长等差数列的长度，用了dp
给一个sorted array of floats, 和三个参数a b c，返回一个array of  sorted f(X) 其中f(x) = a*x^2 + bx + c 用脑海中仅存的数学知识求了个导，变成了一道通过binary search把X分成两半分别求f(x)然后merge two sorted array ，只让写了main function和merge sort部分，没写binary search部分。
利口339 + followup变种 [1[2[3]]] 求 1*1+1*2*2+1*2*3*3
給你一個array, ex.[abc, cde, zab, aa, bb]要你返回[[abc,cde,zab], [aa,bb]],因為被分為同組的每個string可以透過每個字母shift一樣的步數得到相同的string
给定一个有向图，和其中一个node，让输出所有的从这个点出发到终点的路径，终点就是只有指向它的edge, 没有出去的edge。
一个只包含1-9的字符串， 可以添加加号或减号，组成的表达式最后的结果是一个target，输出所有的可能
Given a directed acyclic graph, write a function to compress this graph.
对于一个整数数组，后一个数是前一个的加一或者减一，输出所有local maximum和local minimum
将int a插入一个sorted cyclic linked list，e.g. 将5插入 1 -> 2 -> 4 -> 6 -> 10 ->
给一个array里面装 task 编号， 相同编号task 不可以 在K个单位里相邻，如果相邻，则插入等待的unit 时间 求最少时间。
给一个list of primes，要求print他们所有可能得出的product，每个prime number只能用最多一次。比如given [2,3], return [2, 3, 6]
利口253（meeting room）的变种，room的数量固定是1，问最多能开多少回，我用的dp 
大数相乘 
Guess Word
高频的01矩阵找最左边1所在的列，假装没做过，分析了从O(m*n) 到O(m*logn)到O(m+n)的算法
给一组list，[[2,3], [3,4], [5,6,7], [1,2]]，只要有相同元素的list都可以合并，最后要求返回合并之后的list [[1,2,3,4],[5,6,7]]。当时没想起来用union-find，就先分析了暴力的方法，然后尝试将每个list和元素做mapping，小哥提醒说方向对了
利口3（最长不重复字符的子串）。abccc -> 3
题目是给一大堆邮箱，如何判断哪些邮箱属于同一个人，讨论input和output之后写代码。这个比较费时间。是用union-find解决的
有点像利口450。说有一个balanced BST，treeNode除val外，还有一个值为Mode。mode值不确定，可以为0或非0。mode对树结构无影响。要求删除树中所有mode为0的节点。
LC865的变种 865是二叉树 楼主被问的题应该算是n-ary tree
给一个由数字组成的字符串a，重新排列所有的数字生成一个新的数b，使得b < a 并且 b是所有小于a的组合中最大的。类似于 https://leetcode.com/problems/next-permutation/
 从左到右按列打印binary tree。列相同的情况下，优先打印行数小的node （离root近的node）。同列同行的nodes，先打印哪一个都行。Output to console。
利口297，面试官说如果树很大，递归写法会stack overflow，建议看一下循环写法实现。
利口340（Longest Substring with At Most K Distinct Characters）
利口25（reverse nodes in k-group）
利口987（垂直遍历二叉树）

System design

Design facebook mem cache
设计一个图片分享app， 讨论了怎么实现image feed还有实现拍照获得图片
google photos app
Design Instagram
设计一个景区签到系统，估计类似Yelp，答主答了Geohash 之类的东西
加面的题目是设计单机版memcached
这个不是系统设计，我主要说了底层的设计，hashtable, lru，memory management，怎么处理高并发。可能这个题要看面试官怎么问，这个可以问很深我觉得
Design a tree of sensor
design load balancer
Facebook API News Feed
给一个坐标，返回和它距离k的所有建筑
https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=536531
用户过去七天里听的最多的十首歌
用户每听一首歌，就调用一次API: hit(userID，songID)，设计一个系统，返回用户最近7天听过最多的歌
经常被考的那道搜索自动补充
设计在线text autocomplete，要求reliability，low latency
location based design. 类似yelp
asked to use quad-tree not geo hash)
He said geohash is not accurate, as close points might not have close geohash - I think quad tree also has the same issue
Design a translation/internationalization service for an application/web service like FB.
设计一个类似网页translator。网页的内容需要根据不同的地区翻译成该地区的语言。
follow up - 对于dynamic的内容，如何实现
design leetcode, bottleneck 
discuss where is the bottle neck (compiler and runner), how to optimize (stateless, can be scaled out), queue ..., don't forget to mention security.
设计类似LeetCode的做题网站，每天可以提供若干次竞赛，用户需要注册之后登录然后提交代码，关键是要快速列出ranking page，这也是我工作的一部分，感觉这两轮系统有点为我定制的感觉。
design instagram，面试官说这个是API design 不是system design
要设计的功能包括Post，view, 小头像怎么存，如何like, 如何显示like (liked by xxx and 50 others), 如何评论，如何显示评论，如果网络不好会有什么影响 (还有一些不记得了）
怎么利用几千台机器做个crawler
大概根据我对crawler的理解说了，包括DHT，DB设计，用task queue发布新的url，别的server来pull，还有一些估值什么的。
要求不重复下载，机器间通迅最少
重点是要注意处理URL的hash，不要重复crawl，考察点是Sharding
每台机器bfs去crawl那些倒不是重点
这题我觉得可以用consistent hashing 的思路解决。每台机器都有一个指定的URI，设计一个hash function，能将url和机器的url的hash值平均分布在0~2^64中。每当一个机器获取一个List<url>, 算每个url的hashcode 然后发给指定的机器。所以每个机器中有两个表，一个是hashcode对应每个机器的表，另一个表是已经visited的url的map。至于减少机器之间的通讯，不知道怎么弄。。。
如果在意速度不太在意精确可以试试bloomfilter + 参考mapreduce的partitioner 将类似的url发到指定的计算机上？ lz是怎么答得呢？
一點想法獻醜: 
思路從 BFS 延伸，但最關鍵的點就在於只有一個起點為開始
總不能讓所有機器都從這個起點算起，做去重，這樣會有大量的 conflict
所以我的思路是邊 BFS，某些 neighbor link 就自己算，某些就分配到 nearby computers
然後整個 workload 就會用傳染的方式，讓多台機器並行去算 
但這個方法，可能會有些機器根本就沒用到的狀況
簡單說就是併發度不夠，明明可以更快卻快不起來


為了改善這個算法
我想如果一開始就能知道由起點展開的完整圖，那就可以最佳化
將 graph 平均拆分成幾塊 sub-graph
每個機器去算若干個 sub-graphs，這樣就能最大化利用到每台機器


所以我的想法是:
一開始先用方法一 (某台機器為起點，傳染方式給周遭) 這步來建圖，
1 billion 的 URL，假設長度 100 字元，每個 graph node 跟 edge，假設要 50 Bytes，那麼一個 node 需要 1 billion * (100 + 50)(bytes) = 150 GB
10k 機器，總 memory 為 2500 GB，要拿其中約 150 GB 存 graph，綽綽有餘


接著用方法二來分配 workload 給所有機器並行下載 HTML


但有個問題是分配 workload 這步要怎麼切圖，這部分再請大老們分享一下思路，謝謝


设计怎么通过hashtag找posts，Instagram为例。讲了通常分布式系统的设计，怎么build index，sharding之类的
Facebook live comment system
design ads 投放系统，每个ads有bid，也有target，还有budget，最高的符合target的bid的ad将被推给user，如果还没有超过budget的话
比如budget是不是必须要到了0了马上不deliver这个ad了，还是可以用async的方式减少latency但是可能over deliver，另外target这个地方，一种方式是全扫描，另外就是考虑加index，还有就是利用类似decision tree的方式呢对于这个user所在的分类/group直接返回相应的一组ads（这个方法assume read》》write new ads）不过最基本的就是先分析要求，其次high level end to end的design，然后试图deep dive
design a dropbox/file-sharing system 
日志收集系统
分布式收集，统一处理， 可以参考一下脸熟的Scribe
Type lookahead
设计search infra to return typeahead (mainly focused on the components after tokenization and query understanding, machine learning for ranking is not considered here)
design a photo synchronization app 感觉跟dropbox有点像
sys design, given list of places, find all places within circular distances of requested point at x, y.
这题我回答的是geohash，但对方一直追究geohash角落问题，个人对location理解不深，大家good luck
一个上传/下载系统，支持图片视频之类
设计Facebook News Feed
Design infra and storage for posts with same keywords
Design search bar
Facebook Messanger
Design a distributed block storage
Design Facebook search。比较常规的系统设计题，无非就是如何scale up，如何shard数据，如何handle突然变得popular的search。
knight in an infinite chess board，输出从起点到另一点的最短距离，有obstacle。BFS。
做题，不是原题但类似。给个字符串”123456789“，要求插入”+-*“，使得最后计算数字等于某个给定数字，输出所有满足条件的结果。dfs。写到最后他看起来不太高兴：+12*3和12*3是等价的，要求只输出一个。仔细问了时空复杂度及优化。就做了这一题
design一个http download lib。题目就这么长，具体需求都是自己去问他。我问了文件大小（MB~GB）、主要面向的使用场景（desktop下电影下图片），多job，允许用底层库比如libcurl。我的思路是用一个thread pool去实现，API提供了两种：callback式的和promise/future式的。最后一直扯到thread的scheduling。。（android 题）
设计一个fb食堂的订餐系统，员工和visitor可以从一个食堂order meals，订单的信息需要persistent。按照标准思路答，问下用户量，设计service和schema，比如meal可以存在NoSQL里，订单信息和user信息可以存在SQL里，怎么scale。我觉得可以参考怎么设计Uber Eats（LBS部分除外）
设计一个非死不可员工自己用的订菜外卖系统， 支持两种用户，客人和参观，然后一些预定分发功能等等
设计Calendar。我忘了Event object除了time, user，还有location也很重要。
use cases多了
recurring event
handle event conflict
send user notification ahead of event
实现一个搜索的功能，给定一个query，返回相关的documents。
给一个地点和一个距离，和一大堆places，设计一个service返回这个距离内所有的places，重点focus在如何存储这些places和如何query，以及如何把这么多数据分别存储
设计一个在分布式集群环境下提交job的service
service本身有哪些可能失败的地方，如何解决
service如果提交job失败，然后报错是内存不足，怎么处理
nearest point of interest， geohashing
system design：design a Facebook scale chess game which you can play 1v1 with your friends, you can play multiple games on multiple devices with multiple friends together at the same time, you can undo your last move before the other player takes the move
在資料庫裡不需要存棋盤，只需要跟ＣＨＡＴ一樣存對話就好，比方說：
Game 12345
===========
Msg id 1 | Player A | (0,0) -> (0,1)
Msg id 2 | Player B | (7,0) -> (6,0)
Msg id 3 | Player A | (0,1) -> (0,2)
Msg id 4 | Player A | revert
Msg id 5 | Player A | revert
Msg id 6 | Player B | (6,0) -> (5,0)


因為棋盤跟規則是固定的，LOAD GAME的時候只需要重放操作（無效的操作會被skip，比方說msg id 5），在local/device上cache棋盤和已經播放到哪一步就好。
剩下的就跟chat system 一樣
棋盘就是用8*8的2D array存就行了，应该还可以再压缩点，然后感觉就和Design chat app的套路差不多了。
感觉难点是undo your last move 如何处理race conditions
fb不考OOD的吧，我在想题目说 play multiple games on multiple devices with multiple friends together at the same time, 和device有什么关系，另外是不是要考多线程？
我也觉得跟design chat差不多，差别在一个group只有两个人，但是undo last move感觉不用care race condition，因为轮流turn，一定是你的turn才能undo，既然棋盘都能存了，感觉问题不大？
design the newsfeed product, focus on the API design and front-end integration, need to support web browser, smartphone and feature phone, etc. how to optimise for different devices to have a good user experience?
设计100K的web服务器到10K（1000 QPS）搜索服务器的最佳通讯方式，利用load balancer或者message queue来做。我是第一次onsite system design，直接懵了，恰好自己在做类似的工作，所以直接按照工作经验在找最好的解决方案。快结束的时候才想起来这是面试要多交流、多问问题，但是已经完了，估计国人小哥当时也是懵了，给的数据基本上都没怎么用上。
design friend list online and offline feature
简单的思路：
assumption: 针对的use-case是用户的朋友数量在1~1000个的情况，通常fB也限制朋友的最大数。
1. 在用户初次登录的时候：从server读取在线friend的列表，并更新自己的online friend列表。同时，用户的状态由offline变成online也作为status change的消息push到server。
2. 当用户在线使用FB的时候：朋友都有自己的状态，如果一旦发生status change，把status change作为notification push到server，然后转给自己。例如，我正在使用FB，我的页面上显示了目前在线的friend，如果其中某个朋友下线了，他的状态改变会通过notification push到server再转给我，或者直接notify我的client端。


没FB和系统架构的经验，求拍！
你的思路对头。细节问的很细，你的列表肯定存在缓存里，比如Redis，application server可以直接fetch，比如schema的设计，web server and data access server，DB and Redis具体如何同步。可以zoom in的很细。还有，DAU给我的数据是1 billion的用户。我这没设计过TPS这么高的系统。
感谢你把系统面试drill into这么细的程度。给我进步思考的信息。


背景：我没有设计和开发分布式系统的经验，目前正在看design data-intesnive application那本书。同时在研究了很多论坛的总结帖子和油管的技术会议视频。我没有太多经验，只能根据看这些资料来讲我的理解了。


1. 缓存： 缓存的使用应该是比较直接，用户登录系统之后，web server通过缓存获取自己全部信息（包括自己所有朋友列表），如果缓存没有，就去route to dB获取用户信息。在得到用户自己的朋友列表之后，再去访问另一个用户状态的缓存服务器（假设存在这样一个用户状态缓存服务器）。假设用户自己和朋友的社交网络graph的地域分布特征为本区域和本国家内的用户的connect较密集，国家之间、洲际之间用户的connect较稀疏； 所以这个用户在线状态缓存，主要保存本地区用户状态。同时少量的其他国家、地区的用户在线状态,方便那些具有多国朋友的人查询自己的好友是否在线。


2. Schema的设计： 这里的schema，我理解的是用户是否在线状态的设计，这个应该是in-Mem的schema-less的JSON对象。如果只是为了online/offline这个应用，在设计schema的时候只需要用少数几个attribute，例如： userID，userName，country，Region etc 这样也可以减少内存需求空间。至于脸书这个社交网络平台的其他schema的设计和考虑就比较宽了，设计的内容太多。


3. web server与data access server的同步：不知道需要讨论那些问题。


4. 数据库与缓存（例如Redis）的同步： 如我上面理解的，用户在线状态信息应该全部通过in-mem获取，如果仅仅是确定在线好友状态，不用去访问关系型数据库（RMDB)。如果是其他信息的话，遵从先缓存更新，后RDMS更新的基本原则。当然需要考虑读/写数据的request load来采取具体策略。


5. 1B的用户数量：还是需要考虑分片（sharding）。我前面假设脸书的用户具有很明显地域分布特征。在分片的时候，主要考虑把本国本区域的人的在线状态信息存储在附近的datacenter。


这五个方面我的理解肯定都比较肤浅。望楼主多多指教。也方便活跃版面热烈讨论技术的氛围。
design privacy content 是share 你的post 给specific group you want
design聊天软件 不用考虑group chat和offline notification
design netflix
感谢你把系统面试drill into这么细的程度。给我进步思考的信息。


背景：我没有设计和开发分布式系统的经验，目前正在看design data-intesnive application那本书。同时在研究了很多论坛的总结帖子和油管的技术会议视频。我没有太多经验，只能根据看这些资料来讲我的理解了。


1. 缓存： 缓存的使用应该是比较直接，用户登录系统之后，web server通过缓存获取自己全部信息（包括自己所有朋友列表），如果缓存没有，就去route to dB获取用户信息。在得到用户自己的朋友列表之后，再去访问另一个用户状态的缓存服务器（假设存在这样一个用户状态缓存服务器）。假设用户自己和朋友的社交网络graph的地域分布特征为本区域和本国家内的用户的connect较密集，国家之间、洲际之间用户的connect较稀疏； 所以这个用户在线状态缓存，主要保存本地区用户状态。同时少量的其他国家、地区的用户在线状态,方便那些具有多国朋友的人查询自己的好友是否在线。


2. Schema的设计： 这里的schema，我理解的是用户是否在线状态的设计，这个应该是in-Mem的schema-less的JSON对象。如果只是为了online/offline这个应用，在设计schema的时候只需要用少数几个attribute，例如： userID，userName，country，Region etc 这样也可以减少内存需求空间。至于脸书这个社交网络平台的其他schema的设计和考虑就比较宽了，设计的内容太多。


3. web server与data access server的同步：不知道需要讨论那些问题。


4. 数据库与缓存（例如Redis）的同步： 如我上面理解的，用户在线状态信息应该全部通过in-mem获取，如果仅仅是确定在线好友状态，不用去访问关系型数据库（RMDB)。如果是其他信息的话，遵从先缓存更新，后RDMS更新的基本原则。当然需要考虑读/写数据的request load来采取具体策略。


5. 1B的用户数量：还是需要考虑分片（sharding）。我前面假设脸书的用户具有很明显地域分布特征。在分片的时候，主要考虑把本国本区域的人的在线状态信息存储在附近的datacenter。


这五个方面我的理解肯定都比较肤浅。望楼主多多指教。也方便活跃版面热烈讨论技术的氛围。



Design 总结：
https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=537948
Design messaging system ---- 我觉得这个偏backend, 因为数据量比较大，db设计也比较复杂，感觉后端很有考点。
Design instagram --- 这个我觉得像API design， 因为我觉得和news feed很像。。。
Design news feed API --- 这个肯定是API咯，
Design POI, Design typeahead --- 这两个都用比较特别的data structure来存储数据 （quadtree/geohash, trie), 所以应该偏后端
Design web-crawler --- 这个应该也是后端， 因为本身就不是面对用户的产品。
Design memcache --- 这个应该也是后端
Design search system --- 这个像API， 因为前端蛮多use case的， 觉得API有些讲头
Design internationalization service --- 这个也是API，db没啥可讲。。。。
Design leetcode --- ？？？其实不知道这个题的requirement....
设计一个系统, 返回用户最近7天听过最多的歌 --- 这个也像是backend, 因为use case比较单一，难点在后端
设计一个fb食堂的订餐系统 ---？？？？不知道。。。。







Behavior:

最成功的项目；有跟别人有conflict吗；最不喜欢跟什么样的人work；最喜欢组里的什么人，为什么；有什么失败吗，学到了啥；

个人喜欢套 “举个例子”➕“所以我学到了blah blah”这个模板，建议楼主去看看Dan Croitor的油管，讲得很详细

1) dive deep into most recent project 2) most interesting debug/bug 3) describe most favorite day in your current company（呵呵） 4) how to work across team 5) give me an example you push back 6) example you regret your design 


Culture fit 这轮真的非常重要。 千万要重视起来。 如果是面E5。 一定要尽力往怎么lead project 上靠。 core values 也一定要记牢。

和同事conflict啊，最proud的project啊之类的


e5 one system design
E6，两轮设计

谈经验 谈leadership 谈项目 谈各种soft skills



HR mentioned: 
1. Show leadership in both ppl management and Project scope, in both experience and system design round
2. Clean code with test cases
3. System Design里一定要主动drive这个chat，面试官会很希望你多问多想，他会帮助你narrow down这个题目的components
4. 在system面试里有机会可以dive deep into your domain knowledge. 如果你可以show一些面试官感兴趣但是maybe也没有很懂的东西 很有利于定级

NYC Onsite:
Timeline: coding1 + product design + lunch + background + coding2 + coding3


常规behavior问题，聊proud的项目（问得很细，比如metrics），如何处理和coworker的冲突，等等。这轮并不像recruiter说的会有一题coding。


依然不知道怎么答比较好，最后面试官说：你说了太多detail，我没看出你是怎么从这些经验中学习的，感觉凉了。全程我采用了一个策略：我承认自己虽然目前情商不行，但愿意学习，最后问了他好多这个方面的问题，问他怎么学习这方面的能力的。他啪啪啪地记下来，感觉我的策略奏效了？


她问你有没有过，完成了一件事情后有如释重负的感觉。楼主答的是most challenging，但这里她可能更想要我为什么觉得它很难，以及我的态度。


和同事老板conflict怎么样，同事给你feedback有啥受益的么，老板好么怎么变得更好么，好员工有哪些品质？这个品质你有吗，举个例子啊。这是亚麻学的面试么，脸书也走这一套了。而且要求和亚麻一样，例子要详细，技术细节上要面试官听的懂，而且故事也要make sense，要面试官也同意你的做法是对的。连说45分钟，很累。

why facebook


 one project that you are proud of/Team project/Disagreements/Constructive suggestions from manager 基本是地里常见的


bq 问了most proud project，project business value， internship 喜欢哪点 学到什么， team conflict



常规bq，比较难的大概就是收到最有建设性的feedback是什么


第一轮：bq。能记住的问题有：
1） 介绍骄傲的project。遇到了什么问题。
2） 怎么处理和manager的矛盾
3） 你有role model么？是谁？你觉得你和她的差距是什么？

问简历并且说的很具体。你收到的manager 的最差的feedback是什么。你最喜欢的教授和最讨厌的教授。bq答的不是很好。

bq记得的有先说一个team work的project，然后讲有没有和队友产生分歧以及解决方法，然后最proud的project，然后在team中有没有其他队员之间有冲突或者不engage你是怎么解决的，还有最difficult的队友

希望改进fb哪一点

最proud的项目；收到的positive的feedback是什么；和其他team合作的经历；直接和客户对需求的经历；和别人有分歧的经历；和别人有分歧但其实错在自己的经历；不太agree with的经理的管理方式

Tell me about one of your successful projects.
What would you have done differently in this project?
Tell me the type of person you don't want to work with.
LeetCode刷tag，高频从上到下就够了，我看了之前其他同学的面经总结，高频题都一样，低频的各不相同，低频碰上原题的概率极低，尤其是Uday，几十个人至少也得几十个面试官。所以高频准备好好的就行了。

BQ: 最骄傲的project，为啥骄傲，如何convince others，如果能再做一遍这个project，你会改进什么地方

fav project / biggest challenge as in technology, communication, scale, people, etc / disagreement with manager, peers, ppl from other teams, how to solve at the end / what to expect at next job / least fav in work

bq各种least问题，least internship， least person you work with,

https://www.1point3acres.com/bbs/thread-195416-1-1.html


---


经验：
结合我最近看design的资料，给楼主一点建议吧。


1 着重在thinking process，不要想45分钟给一个完美的解决方案

2 条理要清晰，有big picture，有data modeling，有data flow

3 多从customer角度来考虑，用户要什么知道了，你才好决定你的设计中的取舍

4 不要一开始就挖大坑，从MVP开始，最后时间足够才加新的feature

5. 从最开始简单的架构开始迭代，分析负载上去了之后的bottleneck在哪儿，你要怎么优化。这时候再带入一点customer最关心的是哪里，可以做哪些tradeoff。你做这个技术选型的原因是什么，因为技术mature还是没有SPOF

6 最后一定要review一下系统的scalability，负载x10的话，你的架构是不是线性的加资源进去就好了？

7 架构确定好了之后，最好来个预估要达到这个性能指标，我们需要多少机器以及怎么部署

8 别忘了提一嘴logging monitoring

9 扯一扯service oriented architecture也是极好的

10 把qconf 2012里scaling Pinterest from 0 to billions看几遍，我看了那么多资料里面，最有启发的一篇


总的来说，我不太建议一上去就摆一个高大上的架构，从小到大循序渐进的来，你就可以展现你更多的能力，不光有breadth还有depth。系统设计都是相通的，碰到没见过没准备的不要蒙蔽，毕竟要考察的是思维过程，不是要求你设计一个完美解决方案。



我自己的系统设计套路：
1. 讨论用户是谁
2. 根据用户讨论feature
3. 问一下系统需要handle 的traffic, 问问需不需要进行计算。 面了8次系统设计，只有roblox 要求计算。其他都不要。。。
4. 根据feature讨论系统需要存储和serve哪些data, 这些data用什么存， 讨论sql/nosql/cache/object storage/hdfs 取舍， 巴拉巴拉。。。
5. 根据数据， 设计service。 画图。
6. work through一个use case, 把所有service连起来， 同时修改刚才画好的图。 比如 做uber eats, 讨论用户要order 一个食物，到餐馆接到订单， 到司机接到订单。。。。
7. 讨论use case细节， 比如 uber eats司机进入某个区域怎么识别啊， cache里怎么存啊。面试官全程都会drive你的design的， 不会丢你在那里自言自语。
8. 面试官会问， 某些环节挂掉了，怎么处理。 无非就是1. 要么replica， master slave, active-passive 或者 2.周期存snapshot 在磁盘上，然后存action log... 挂了可以重新恢复。。。
9. 一些环节怎么scale... multi instance, partition 这些呗。。 偶尔说说service mesh...

onsite面试经验：
- 我面的8次系统设计体验都很好， 面试官会drive design全程， 他会不停的问你小问题带你走。 当然他也会根据你的设计不停的提出小问题。 交流交流交流啦～
- 有的时候面试官会质疑你的设计。 此时有两种解法， 1. 解释自己为什么这么做， 让面试官认同你的做法 2. 想一想是不是哪里做的不对， 面试官是不是再给你hint，要换一种设计。  具体情况具体分析啦。。。
- 我至少3次面试，解释了consistent hash 和 virtual node 怎么回事。。  不懂得朋友， 学学看哈。。
- 每个公司的系统设计面试题都蛮固定的。 提前好好刷以下面经，准备准备。




背景：我没有设计和开发分布式系统的经验，目前正在看design data-intesnive application那本书。同时在研究了很多论坛的总结帖子和油管的技术会议视频。我没有太多经验，只能根据看这些资料来讲我的理解了。


1. 缓存： 缓存的使用应该是比较直接，用户登录系统之后，web server通过缓存获取自己全部信息（包括自己所有朋友列表），如果缓存没有，就去route to dB获取用户信息。在得到用户自己的朋友列表之后，再去访问另一个用户状态的缓存服务器（假设存在这样一个用户状态缓存服务器）。假设用户自己和朋友的社交网络graph的地域分布特征为本区域和本国家内的用户的connect较密集，国家之间、洲际之间用户的connect较稀疏； 所以这个用户在线状态缓存，主要保存本地区用户状态。同时少量的其他国家、地区的用户在线状态,方便那些具有多国朋友的人查询自己的好友是否在线。


2. Schema的设计： 这里的schema，我理解的是用户是否在线状态的设计，这个应该是in-Mem的schema-less的JSON对象。如果只是为了online/offline这个应用，在设计schema的时候只需要用少数几个attribute，例如： userID，userName，country，Region etc 这样也可以减少内存需求空间。至于脸书这个社交网络平台的其他schema的设计和考虑就比较宽了，设计的内容太多。


3. web server与data access server的同步：不知道需要讨论那些问题。


4. 数据库与缓存（例如Redis）的同步： 如我上面理解的，用户在线状态信息应该全部通过in-mem获取，如果仅仅是确定在线好友状态，不用去访问关系型数据库（RMDB)。如果是其他信息的话，遵从先缓存更新，后RDMS更新的基本原则。当然需要考虑读/写数据的request load来采取具体策略。


5. 1B的用户数量：还是需要考虑分片（sharding）。我前面假设脸书的用户具有很明显地域分布特征。在分片的时候，主要考虑把本国本区域的人的在线状态信息存储在附近的datacenter。


这五个方面我的理解肯定都比较肤浅。望楼主多多指教。也方便活跃版面热烈讨论技术的氛围。

