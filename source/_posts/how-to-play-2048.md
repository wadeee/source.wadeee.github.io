---
layout: post
title: 2048这游戏怎么玩
published: true
date: 2014-06-08
---

> 发一篇陈年旧文，发现还是挺有趣的，就像翻开了尘封多年的日记一样。

<!--more-->

![实力图][1]


由于之前曾夸下海口要为几个极度渴望游戏秘诀的同学写个通关秘籍，所以此时此刻出现了这么一篇蛋疼的东西。其实这个游戏跟智力也没太大关系，同学们做消遣玩玩也就过了。如欲深究，我们就来谈谈。

## 从头开始

不能马虎，所以我们一切从头讲起。

![2]

最开始玩这个游戏，当然是上左下右循环乱搞一番，那时候玩出个1024牛逼哄哄，简直觉得自己智商已经封顶。不过多玩两次开始发现了一点规律，我开始把大的数堆在右下角（我知道大家习惯都不一样，堆在哪里都是可以的，自己看的舒服就行），遵循同排的从左往右递增，同列的从上往下递增，只要保持这样的排列方式，可以比较轻松的到达512或者更高。这是规律最初雏形，不再是毫无目的的瞎玩。

![3]

接下来这个方法很快出山了，就是当前大众所熟知的秘籍走法。我也不知道要怎么说这个规律和前面所说的那种规律有什么具体的联系，但起码我是从前面的方法过渡来的。

![4]

在慢慢熟悉了这个方法之后我开始嚣张了，开始觉得这个游戏太弱了，2048简直太弱，开始挑战4096但屡屡失败。就在当我觉得4096无望的时候，伟大的昌老师出现了，他在空间发了一个他玩到4096的照片，这时候我们就可以看到那种在泡沫剧里面常见的剧情，我被昌老师的一张图片所激励，废寝忘食地玩，最后成功完成4096大业。

![5]

 玩到4096后我继续玩，刷了个比较高的分数拿出来炫耀一番，不要脸地得到了各方奖赏，也就是在被奖赏的同时贸然答应了同学要把秘籍写出来，简直是太不要脸了。

不过对这个游戏的研究还没结束，有人说这个游戏的精髓就在这个蛇形的构造，其实不对，这个游戏真正的精髓在于懂得取舍和对付死局的能力。为什么我这么说呢，其实简单的来说，就是区分你玩10盘可以玩出2次4096，还是8次4096。废话不多说，我们开始来进入正题。

 
## 取舍

关于取舍，就是我们走一些关键的步要顾大局，其实每一步都要顾大局，强迫症不改则乱大谋，这个必须要改，接下来我们举个例子吧。

![6]

这是最简单的例子，当出现这种局的时候你会怎么做呢？解这种局最忌讳的是因小失大，因为心中总是惦记着下面两个小的数，怎么也放不下这两个格子，以至于一直要急于把这两个小数解决。

![7]

其实纵观大局，只不过是失去了两个格子，我们称其为坏点，暂时放弃即可，用前三行解出一个512与底层的512会和即可解放，如若不放弃，就容易酿成这样的错误。

![8]

走到这一步，你可能解不了底层这个2（其实这个例子是可以的，但是很多时候你并不这么幸运），这个时候你唯一的方法还是放弃他，而此时你又多了一个坏点，这时候你只有11格可以解出一个512。可能少了一格你觉得差不太多，但如果你多了一格，你会多一份希望，坏点多一个，你成功的可能性就会低一点。

![9]

有了上面的经验，你应该明白遇到这种情况要怎么做了吧！

![10]

不过肯定有人会吐槽，用11格解一个512还是可以的。确实，用11格解一个512可能并不能难倒你，但是当数变更大的时候，你一定要做出正确的抉择，要不你连理论上的可能都没了。只是为了一个毫无用处的坏点让你输了整盘游戏，这是最不值得的输法，定要记住。

## 死局

接下来我们讲讲什么是死局。

![11]

其实准确来说，死局只是我给他取的一个难听的名字，因为死局并不一定意味这这盘已经输了，但是如果你不懂处理，那百分百是输了。

![12]

我统称这种左右下三个操作完全不能动弹的局为死局，其实很无奈，死局令我们破坏已定的阵型，最下面一行的数一般是最大的，如果被顶上去后果就是很难了。

![13]

当然这种时候你要学聪明了，看上面这种情况，我们肯定不罢休，谁都不想把最大的1024顶到第二排。所以这个时候肯定是往左操作。

![14]

如果运气好，顶起的数比较小，我们就可以往下划了，就比如上图，你需要做的就是将128和256顶起。如果这个2不是刷在256,下面而是512下面，这时候你可以选择再赌一把或者放弃，这取决于你对自己实力的评估，如果你觉得这个512你可以解出1024来你可以放弃继续赌，如果你确定你一定不可能解决这个512，那你一定要继续赌下去，因为对你来说这个512太难了，还不如赌一把来的现实。

![15]

而当你遇到这样的死局，你的选择可能就不多了，但是如果你觉得并非无解，那你就要做好取舍，依旧有机会成功。

![16]
 
这种局也是容易碰到的，他有一定的几率会让我们接下来直接带来一个坏点。因为我们不能向右和向下划，所以我们只能向左划（别跟我说向上划，那样要冒更大的风险）。

![17]

相信在通往4096的过程中你会经常遇到这样的情况，这样的情况有两种解决方法，第一就是直接放弃这个格子，对于这一局，在完成4096前它是一个好不了的坏点。如果你有信心用剩下的15格解出一个4096，那么这个坏点就彻底放弃吧。还有一个办法，就是在逼不得已的时候才用的，就是左右划动（其实适当的时候也可以选择向下划动），直到能将大数固定住，且其他位置留出的空余足够。

![18]

这是一个不错的例子，这个时候你就可以选择向上划。

![19]

在这个例子中，你有百分之七十五的机会可以顺利逃出此劫，只要数字不刷在感叹号所在位置，就可以顺利解除危机。在两个最大数的危机解除之后，你也要仔细规划让相对较大的数回归底边，比如128，我们要尽可能在底边解出256或者512的时候让他顺利回归，如若让它在在第三层慢慢变成一个256或者更大的数，底面就会出现一个甚至两个坏点。

当然我们都不希望遇到这些死局，那么有没有希望去避开这些死局呢？答案肯定是有的，不过不是绝对能够避开，只是尽最大努力去避开。

![20]

比如这种情况，如果想都不想就往下划，那就容易酿成大错，变成下面这种情况->

![21]

按照概率来说，遇到这个情况的概率是10%（出现2 的概率为90%，而出现在该位置的概率为1/9），虽然这个概率很小，但是我们要杜绝这样的情况就要时刻注意防止这样的情况发生，因为谁都不想时时刻刻去处理死局。

![22]

如果你不去扣细节，其实只要不遇到死局就可以继续，这就要大量消耗人品了，但如果你好好地注意细节，更多时候你靠的是实力，而非人品，你可以得到的分数就更高。就这个例子吧，如果你手快选择了向左划，那么你承受的风险就要比向下划更大，因为向下滑的话只有一个格子会迫使你下一步要冒险向右划，而如果你向左划，则是有两个格子会迫使你冒险。将风险降到最低，你才有更大几率玩到高分。

![23]

如果你刚刚不小心让情况发展到这个地步，没关系，我们完全可以承受，这个时候要怎么抉择呢？有的聪明的孩子肯定会说这32和256这两格已经是坏点了。是的，他们确实是坏点，但我们可以减少坏点。

![24]

虽然刚刚的128是在中间，但是我们此时将他收至最右边，这个时候我们只要放弃最底层的256，接下来的目标就是用前三层解出一个512即可。

其实我们会发现，最大数在中间两格的时候经常会出现坏点，所以，为了保护好整个阵型不受移动的影响，我们要每次都要尽快的保证正在奋斗的这一层稳固。

![25]

如果遇到以上情况你会怎么选择？如果往下划就是在给自己找坑，整个第三层层都会变的不稳固，正确的选择是向右边划（向左边划有一定风险，但并不大），然后接下来左右滑动都可以。

![26]

只要这个16的左边出现一个数，你就可以如愿以偿地和第三层的16解出一个32而并不影响第三层的稳定性。

![27]

这样是不是很管用呢，相信你能从这里体会到精髓。

## 总结

![28]

说到这里你应该很快就会明白前两行你能解出的最大数对于你成功的重要性了，我的最高纪录是用前两行解出一个128，这是相当难的，这个就需要你的经验去解决了。如果你没有把握你就应该提前收掉前面的数，比如上面这个图，如果有能力你可以解出一个64去收一个4096，玩的再差我相信你也能凑个16去收，至于收16还是收32，这个你自己掂量着点，因为有时候你凑出一个16但它位置不在最右边，此时一个32才能救你，这些都要自己去慢慢琢磨。

![29]

这是我的最高纪录了，只能说在最后时刻人品没有爆发，只要其中随便一个2换成4或者4换成2我就有信心凑出个16384。这里面有一个坏点是32，其实当时我可以选择让他成为坏点还是留下他的，但是当时第二层有32并且不是在最右边，而格子又太挤了，我毅然地舍弃了他，决定要用两层解出128，但最后事与愿违，只能停留在这里了。希望大家满意，不过还有很多细节的东西我忘了，如果有补充的可以告诉我我可以加上去的。

## 声明

本文由 [Wade\|陈伟城][30] 首发自 [QQ空间][31]

转载请保留以上链接


[1]: /images/posts/how-to-play-2048/1.jpg
[2]: /images/posts/how-to-play-2048/2.jpg
[3]: /images/posts/how-to-play-2048/3.jpg
[4]: /images/posts/how-to-play-2048/4.jpg
[5]: /images/posts/how-to-play-2048/5.jpg
[6]: /images/posts/how-to-play-2048/6.jpg
[7]: /images/posts/how-to-play-2048/7.jpg
[8]: /images/posts/how-to-play-2048/8.jpg
[9]: /images/posts/how-to-play-2048/9.jpg
[10]: /images/posts/how-to-play-2048/10.jpg
[11]: /images/posts/how-to-play-2048/11.jpg
[12]: /images/posts/how-to-play-2048/12.jpg
[13]: /images/posts/how-to-play-2048/13.jpg
[14]: /images/posts/how-to-play-2048/14.jpg
[15]: /images/posts/how-to-play-2048/15.jpg
[16]: /images/posts/how-to-play-2048/16.jpg
[17]: /images/posts/how-to-play-2048/17.jpg
[18]: /images/posts/how-to-play-2048/18.jpg
[19]: /images/posts/how-to-play-2048/19.png
[20]: /images/posts/how-to-play-2048/20.jpg
[21]: /images/posts/how-to-play-2048/21.jpg
[22]: /images/posts/how-to-play-2048/22.jpg
[23]: /images/posts/how-to-play-2048/23.jpg
[24]: /images/posts/how-to-play-2048/24.png
[25]: /images/posts/how-to-play-2048/25.png
[26]: /images/posts/how-to-play-2048/26.png
[27]: /images/posts/how-to-play-2048/27.png
[28]: /images/posts/how-to-play-2048/28.png
[29]: /images/posts/how-to-play-2048/29.jpg
[30]: https://wadeee.github.io "Wade blog"
[31]: https://user.qzone.qq.com/363914451/blog/1402219431
