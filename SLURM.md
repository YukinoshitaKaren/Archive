**一个提交脚本：**

```text
#!/bin/bash

#SBATCH -n 1 # 指定核心数量
#SBATCH -N 1 # 指定node的数量
#SBATCH -t 0-5:00 # 运行总时间，天数-小时数-分钟， D-HH:MM
#SBATCH -p debug # 提交到哪一个分区
#SBATCH --mem=2000 # 所有核心可以使用的内存池大小，MB为单位
#SBATCH -o myjob.o # 把输出结果STDOUT保存在哪一个文件
#SBATCH -e myjob.e # 把报错结果STDERR保存在哪一个文件
#SBATCH --mail-type=ALL # 发送哪一种email通知：BEGIN,END,FAIL,ALL
#SBATCH --mail-user=netid@nyu.edu # 把通知发送到哪一个邮箱
#SBATCH --constraint=2630v3  # the Features of the nodes, using command " showcluster " could find them.
#SBATCH --gres=gpu:n # 需要使用多少GPU，n是需要的数量
#SBATCH -t 1-00:00:00         # 最长运行时间是 1 天
 

runYourCommandHere
```

**监控任务：**

```text
squeue -u yourUserName
```

```text
squeue -j JobID
```

查询特定任务细节

```text
sacct -j JobID
```

```text
squeue | grep 用户名
```

**取消作业**

```text
scancel <JobID>
```

**查看集群**

```text
sinfo
```

```text
cinfo -p <分区名> occupy-reserved
```

**示例**

![QQ图片20230404131258](C:\Users\36475\Desktop\沉思录\picture\QQ图片20230404131258.png)

**指定或排除特定节点**

```text
-w $hostname：指定任务运行节点，注意对应节点必须在指定的分区内

-x $hostname：排除节点，任务将不往指定节点进行调度，一般用于避开故障或异常节点
```

#### 节点状态查看

![img](https://ask.qcloudimg.com/http-save/7611843/am75nvvkxt.png?imageView2/2/w/2560/h/7000)

- PARRITION：节点所在分区
- AVAIL：分区状态，up标识可用，down标识不可用
- TIMELIMIT：程序运行最大时长，infinite表示不限制，如果限制格式为days-houres:minutes:seconds
- NODES：节点数
- NODELIST：节点名列表
- STATE：节点状态，可能的状态包括：
  - allocated、alloc ：已分配
  - completing、comp：完成中
  - down：宕机
  - drained、drain：已失去活力
  - fail：失效
  - idle：空闲
  - mixed：混合，节点在运行作业，但有些空闲CPU核，可接受新作业
  - reserved、resv：资源预留
  - unknown、unk：未知原因
  - 如果状态带有后缀*，表示节点没有响应

#### 分区信息查看

![img](https://ask.qcloudimg.com/http-save/7611843/l46lowfd3k.png?imageView2/2/w/2560/h/7000)

- DisableRootJobs:不允许root提交作业
- Maxtime：最大运行时间
- LLN：是否按最小负载节点调度
- Maxnodes：最大节点数
- Hidden：是否为隐藏分区
- Default：是否为默认分区
- OverSubscribe：是否允许超时
- ExclusiveUser：排除的用户

#### 作业信息查看

![img](https://ask.qcloudimg.com/http-save/7611843/5fjjv9zw1f.png?imageView2/2/w/2560/h/7000)

- JOBID：作业号
- PARITION：分区名
- NAME：作业名
- USER：用户名
- ST：状态，常见的状态包括：
  - PD、Q：排队中 ，PENDING
  - R：运行中 ，RUNNING
  - CA：已取消，CANCELLED
  - CG：完成中，COMPLETIONG
  - F：已失败，FAILED
  - TO：超时，TIMEOUT
  - NF：节点失效，NODE FAILURE
  - CD：已完成，COMPLETED

#### 作业信息查看

![img](https://ask.qcloudimg.com/developer-images/article-audit/7611843/mwrpjo33kg.png?imageView2/2/w/2560/h/7000)

#### 批处理模式提交作业

1.用户编写作业脚本

2.提交作业

3.作业排队等待资源分配

4.在首节点加载执行作业脚本

5.脚本执行结束，释放资源

6.用户在输出文件中查看运行结果

- 作业脚本为文本文件，首行一“#!”开头，指定解释程序
- 脚本中可通过srun加载计算任务
- 一个作业可包含多个作业步
- 脚本在管理节点上提交，实际在计算节点上执行
- 脚本输出写到输出文件中

以下是一些常见的作业资源需求参数，使用`#SBATCH -xx xxx`的方式写入脚本中即可

```
-J,--job-name：指定作业名称 
-N,--nodes：节点数量 
-n,--ntasks：使用的CPU核数 
--mem：指定每个节点上使用的物理内存 
-t,--time：运行时间，超出时间限制的作业将被终止 
-p,--partition：指定分区 
--reservation：资源预留 
-w,--nodelist：指定节点运行作业 
-x,--exclude：分配给作业的节点中不要包含指定节点 
--ntasks-per-node：指定每个节点使用几个CPU核心 
--begin：指定作业开始时间 
-D，--chdir：指定脚本/命令的工作目录
```

