这是自己创建的首页，可自定义内容，具体参考：https://zhuanlan.zhihu.com/p/366761432
使用方法：
source目录下新建的index.md文档将作为自定义的主页，因此，我们需要屏蔽掉系统默认的主页。
屏蔽掉系统默认的主页的方法是修改_config.yml配置文件的index_generator项为一个无效值。

要想首页显示文章摘要，可以参考https://blog.csdn.net/yueyue200830/article/details/104470646
其实就是在文章中加入description就行。
例如：
---
title: 让首页显示部分内容
date: 2020-02-23 22:55:10
description: 这是显示在首页的概述，正文内容均会被隐藏。
---
