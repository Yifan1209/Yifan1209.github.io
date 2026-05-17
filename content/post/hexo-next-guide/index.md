---
title: How to Use Hexo and NexT Theme
date: 2023-11-30
categories:
  - Legacy
tags:
  - hexo
  - static-site
description: A setup guide for Hexo + NexT — the stack this blog was originally built on before migrating to Hugo + Stack.
---

> This blog has since migrated to Hugo + Stack theme. Keeping this as a record of the old setup.

## Hexo 基本命令

```bash
hexo init          # 初始化项目
hexo new "标题"    # 新建文章
hexo server        # 本地预览 http://localhost:4000
hexo generate      # 生成静态文件
hexo deploy        # 部署
```

## 连接 GitHub Pages

在 `_config.yml` 中配置：

```yaml
deploy:
  type: git
  repository: git@github.com:yourname/yourname.github.io.git
  branch: master
```

安装部署插件并部署：

```bash
npm install hexo-deployer-git --save
hexo clean && hexo g && hexo d
```

## Front-Matter

```yaml
---
title: 文章标题
date: 2023-11-30
categories:
  - 分类名
tags:
  - tag1
---
```

## 添加 categories / tags 页面

```bash
hexo new page categories
hexo new page tags
```

分别在生成的 `index.md` 中添加 `type: "categories"` 和 `type: "tags"`。
