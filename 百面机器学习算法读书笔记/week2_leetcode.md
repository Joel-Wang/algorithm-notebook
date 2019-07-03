# Leetcode算法题：第二周

## **问题**

二分（题号：69）：https://leetcode.com/problems/sqrtx/description/

分治（题号：241）：https://leetcode.com/problems/different-ways-to-add-parentheses/description/

链表（题号：160）：https://leetcode.com/problems/intersection-of-two-linked-lists/description/

哈希表（题号：1）：https://leetcode.com/problems/two-sum/description/

字符串（题号：242）：https://leetcode.com/problems/valid-anagram/description/

栈和队列（题号：232）：https://leetcode.com/problems/implement-queue-using-stacks/description/

## 解答

#### [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

*牛顿法，二分法*

实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

示例 1:

输入: 4
输出: 2
示例 2:

输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。

**思路**
牛顿法，$r_{n+1}=r_n-f(r_n)/f'(r_n)$ 从r=x迭代到r<=x/r, 对于本题为x=sqrt(r), 即r^2=x，则f(r)=r^2-x。

此时x视作常数，r为变量，f'(r)=2*r，则牛顿法更新式子为，r=r-(r^2-x)/2*r，化简为r=(r+x/r)/2，为了防止溢出写作r=b+(r-b)/2, b=x/r。由于是从r=x开始迭代，可知x的平方根之前一定有r>x/r，将此作为条件

**代码**

```c++
class Solution {
public:
    int mySqrt(int x) {
        //牛顿法的收敛速度最快
        if(x==0 ||x==1) return x;
        int r=x;
        int b=x/r;
        while(r>b){
            r=b+(r-b)/2;
            b=x/r;
        }
        return r;
    }
};
```



#### [241. 为运算表达式设计优先级](https://leetcode-cn.com/problems/different-ways-to-add-parentheses/)

*分治算法*

给定一个含有数字和运算符的字符串，为表达式添加括号，改变其运算优先级以求出不同的结果。你需要给出所有可能的组合的结果。有效的运算符号包含 +, - 以及 * 。

示例 1:

输入: "2-1-1"
输出: [0, 2]
解释: 
((2-1)-1) = 0 
(2-(1-1)) = 2

示例 2:

输入: "2*3-4*5"
输出: [-34, -14, -10, -10, 10]
解释: 
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
(((2*3)-4)*5) = 10

**思路** 

分治法，以操作符为分割点，计算左右两边所有的可能结果，储存在left数组和right数组当中，然后对left,right 数组的值两两相加得到最终的结果数组。对于每个左右子分支执行相同的操作，当当前的输入没有操作符，则返回当前input对应的数字（以list格式）。即以操作符作为二叉树的根节点，相当于以某个操作符作为二叉树的根节点，然后递归的求左右子树不同的组合。

**代码** 

```python
class Solution(object):
    def diffWaysToCompute(self, input):
        """
        :type input: str
        :rtype: List[int]
        """
        #一个函数calc做运算，
        #一个字典memo记录已经有的结果,key为输入字符串，value为所有计算结果组合
        #遍历input,以一个运算符为界，对左右两边进行计算，得到所有组合后返回，并进行排列组合计算
        #递归边界为input为数字
        if input.isdigit():
            return [int(input)]
        memo={}
        res=[]
        if input in memo:
            return memo[input]
        for i in range(len(input)):
            op=input[i]
            if op not in "+-*":
                continue
                
            left=self.diffWaysToCompute(input[:i])
            right=self.diffWaysToCompute(input[i+1:])
            for num1 in left:
                for num2 in right:
                    res.append(self.calc(num1,num2,op))
        memo[input]=res
        return res
    
    def calc(self,num1,num2,op):
        if op=="+":
            return num1+num2
        elif op=="-":
            return num1-num2
        else:
            return num1*num2
    
```

#### [160. 相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

*双指针法*

编写一个程序，找到两个单链表相交的起始节点。

示例 1：

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_1.png)

输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

**思路**

双指针方法,指针pA先遍历A, 当A为NULL到达末尾时转到headB，指针pB则先遍历B, 到达NULL转到A, 同时NULL则结束，指向相同节点则为交点；

![img](https://pic.leetcode-cn.com/e86e947c8b87ac723b9c858cd3834f9a93bcc6c5e884e41117ab803d205ef662-%E7%9B%B8%E4%BA%A4%E9%93%BE%E8%A1%A8.png)

**代码**

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode *pA=headA, *pB=headB;
        if(pA==NULL || pB==NULL) return NULL;
        //两个指针，方法为交叉遍历，当同时指向NULL时结束，nodesA+nodesB=n，则遍历n个节点
        //time O(n) space O(1)
        while(pA!=NULL || pB!=NULL){
            if(pA==NULL) pA=headB;
            if(pB==NULL) pB=headA;
            if(pA==pB) return pA;
            
            pA=pA->next;
            pB=pB->next;
        }
        return NULL;
    }
};
```



#### [1. 两数之和](https://leetcode-cn.com/problems/two-sum/)

*哈希表*

给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。

示例:

给定 nums = [2, 7, 11, 15], target = 9

因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]

**思路**

采用一个哈希表存储目标数的下标，time O(n), space O(n)

ps: 虽然下标有0但是检测是否存在目标数的条件为m[b]>0，这是因为，数组是不重复的且遍历方向是从0开始的，因此另一个数一定在nums[i]的右侧，所以不可能为0

**代码**

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> m;
        for(int i=0;i<nums.size();i++){
            m[nums[i]]=i;
        }
        for(int i=0;i<nums.size();i++){
            int b=target-nums[i];
            int j=m[b];

            if(j>0 && i!=j){
                return {i,j};
            }
        }
        return {};
    }
};
```



#### [242. 有效的字母异位词](https://leetcode-cn.com/problems/valid-anagram/)

*哈希表*

给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

示例 1:

输入: s = "anagram", t = "nagaram"
输出: true
示例 2:

输入: s = "rat", t = "car"
输出: false
说明:
你可以假设字符串只包含小写字母。

进阶:
如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？

**思路**

排序，但排序的时间复杂度nlogn；采用哈希表可以O(n)复杂度，空间复杂度都为常数级别

**代码1：** 两个哈希表

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        vector<int> hash_s(26,0);
        vector<int> hash_t(26,0);
        for(char c:s){
            hash_s[c-'a']++;
        }
        for(char c:t){
            hash_t[c-'a']++;
        }
        for(int i=0;i<26;i++){
            if(hash_s[i]!=hash_t[i]) return false;
        }
        return true;
    }
};
```

**代码2** 1个哈希表

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        int hash[26]={0};
        for(char c:s){
            hash[c-'a']++;
        }
        for(char c:t){
            hash[c-'a']--;
        }
        for(int cnt:hash){
            if(cnt!=0) return false;
        }
        return true;
    }
};
```



#### [232. 用栈实现队列](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

*栈与队列*

使用栈实现队列的下列操作：

push(x) -- 将一个元素放入队列的尾部。
pop() -- 从队列首部移除元素。
peek() -- 返回队列首部的元素。
empty() -- 返回队列是否为空。
示例:

MyQueue queue = new MyQueue();

queue.push(1);
queue.push(2);  
queue.peek();  // 返回 1
queue.pop();   // 返回 1
queue.empty(); // 返回 false
说明:

你只能使用标准的栈操作 -- 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。
假设所有操作都是有效的 （例如，一个空的队列不会调用 pop 或者 peek 操作）。

**代码** 

```c++
class MyQueue {
public:
    /** Initialize your data structure here. */
    MyQueue() {
        while(!s1.empty())
            s1.pop();
        while(!s2.empty())
            s2.pop();
        front=0;
    }
    /** Push element x to the back of queue. */
    void push(int x) {
        if(s1.empty()) front=x;
        s1.push(x);
    }
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        if(s2.empty()){
            while(!s1.empty()){
                s2.push(s1.top());
                s1.pop();
            }
        }
        int top=s2.top();
        s2.pop();
        return top;
    }
    
    /** Get the front element. */
    int peek() {
        if(!s2.empty())
            return s2.top();
        return front;
    }
    /** Returns whether the queue is empty. */
    bool empty() {
        if(s1.empty()&&s2.empty())
            return true;
        else
            return false;
    }
private:
    stack<int> s1;
    stack<int> s2;
    int front;
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
```

