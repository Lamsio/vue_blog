(window.webpackJsonp=window.webpackJsonp||[]).push([[52],{378:function(s,n,a){"use strict";a.r(n);var t=a(4),e=Object(t.a)({},(function(){var s=this,n=s._self._c;return n("ContentSlotsDistributor",{attrs:{"slot-key":s.$parent.slotKey}},[n("h2",{attrs:{id:"虚拟主机与域名解析"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#虚拟主机与域名解析"}},[s._v("#")]),s._v(" 虚拟主机与域名解析")]),s._v(" "),n("ul",[n("li",[s._v("域名、dns、ip地址的关系")]),s._v(" "),n("li",[s._v("浏览器、Nginx与http协议")]),s._v(" "),n("li",[s._v("虚拟主机原理")]),s._v(" "),n("li",[s._v("域名解析与泛域名解析实战")]),s._v(" "),n("li",[s._v("域名解析相关企业项目实战技术架构\n"),n("ul",[n("li",[s._v("多用户二级域名")]),s._v(" "),n("li",[s._v("短网址")]),s._v(" "),n("li",[s._v("HTTP DNS")])])]),s._v(" "),n("li",[s._v("Nginx中的虚拟主机配置")])]),s._v(" "),n("h4",{attrs:{id:"浏览器、nginx与http协议"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#浏览器、nginx与http协议"}},[s._v("#")]),s._v(" 浏览器、Nginx与HTTP协议")]),s._v(" "),n("p",[n("a",{attrs:{href:"https://www.bilibili.com/video/BV1yS4y1N76R?p=13&spm_id_from=pageDriver",target:"_blank",rel:"noopener noreferrer"}},[s._v("【尚硅谷】2022版Nginx教程（nginx入门到亿级流量）_哔哩哔哩_bilibili"),n("OutboundLink")],1)]),s._v(" "),n("h4",{attrs:{id:"多二级域名"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#多二级域名"}},[s._v("#")]),s._v(" 多二级域名")]),s._v(" "),n("p",[s._v("有时，我们需要建立不同的虚拟主机管理不同的网站请求，因此我们可以创建多个虚拟主机进行管理。\n值得注意的是"),n("code",[s._v("server_name")]),s._v("是可以接受IP、域名、甚至正则表达式的")]),s._v(" "),n("div",{staticClass:"language-txt line-numbers-mode"},[n("pre",{pre:!0,attrs:{class:"language-txt"}},[n("code",[s._v("# 虚拟主机  \nserver {  \n    # 监听端口号  \n    listen       80;  \n    # 服务器IP  \n    server_name  localhost;  \n  \n    location / {  \n  \n        # 页面资源的存放目录，这里是../html下  \n        root   /www/www;  \n        index  index.html index.htm;  \n    }  \n    # 当响应是500 502 503 504时 返回50x.html页面  \n    error_page   500 502 503 504  /50x.html;  \n    # 当用户访问50x.html找不到时，会去../html目录下找  \n    location = /50x.html {  \n        root   html;  \n    }  \n}  \n# 虚拟主机2  \nserver {  \n    # 监听端口号  \n    listen       88;  \n    # 服务器IP  \n    server_name  localhost;  \n  \n    location / {  \n  \n        # 页面资源的存放目录，这里是../html下  \n        root   /www/vod;  \n        index  index.html index.htm;  \n    }  \n    # 当响应是500 502 503 504时 返回50x.html页面  \n    error_page   500 502 503 504  /50x.html;  \n    # 当用户访问50x.html找不到时，会去../html目录下找  \n    location = /50x.html {  \n        root   html;  \n    }  \n}\n")])]),s._v(" "),n("div",{staticClass:"line-numbers-wrapper"},[n("span",{staticClass:"line-number"},[s._v("1")]),n("br"),n("span",{staticClass:"line-number"},[s._v("2")]),n("br"),n("span",{staticClass:"line-number"},[s._v("3")]),n("br"),n("span",{staticClass:"line-number"},[s._v("4")]),n("br"),n("span",{staticClass:"line-number"},[s._v("5")]),n("br"),n("span",{staticClass:"line-number"},[s._v("6")]),n("br"),n("span",{staticClass:"line-number"},[s._v("7")]),n("br"),n("span",{staticClass:"line-number"},[s._v("8")]),n("br"),n("span",{staticClass:"line-number"},[s._v("9")]),n("br"),n("span",{staticClass:"line-number"},[s._v("10")]),n("br"),n("span",{staticClass:"line-number"},[s._v("11")]),n("br"),n("span",{staticClass:"line-number"},[s._v("12")]),n("br"),n("span",{staticClass:"line-number"},[s._v("13")]),n("br"),n("span",{staticClass:"line-number"},[s._v("14")]),n("br"),n("span",{staticClass:"line-number"},[s._v("15")]),n("br"),n("span",{staticClass:"line-number"},[s._v("16")]),n("br"),n("span",{staticClass:"line-number"},[s._v("17")]),n("br"),n("span",{staticClass:"line-number"},[s._v("18")]),n("br"),n("span",{staticClass:"line-number"},[s._v("19")]),n("br"),n("span",{staticClass:"line-number"},[s._v("20")]),n("br"),n("span",{staticClass:"line-number"},[s._v("21")]),n("br"),n("span",{staticClass:"line-number"},[s._v("22")]),n("br"),n("span",{staticClass:"line-number"},[s._v("23")]),n("br"),n("span",{staticClass:"line-number"},[s._v("24")]),n("br"),n("span",{staticClass:"line-number"},[s._v("25")]),n("br"),n("span",{staticClass:"line-number"},[s._v("26")]),n("br"),n("span",{staticClass:"line-number"},[s._v("27")]),n("br"),n("span",{staticClass:"line-number"},[s._v("28")]),n("br"),n("span",{staticClass:"line-number"},[s._v("29")]),n("br"),n("span",{staticClass:"line-number"},[s._v("30")]),n("br"),n("span",{staticClass:"line-number"},[s._v("31")]),n("br"),n("span",{staticClass:"line-number"},[s._v("32")]),n("br"),n("span",{staticClass:"line-number"},[s._v("33")]),n("br"),n("span",{staticClass:"line-number"},[s._v("34")]),n("br"),n("span",{staticClass:"line-number"},[s._v("35")]),n("br"),n("span",{staticClass:"line-number"},[s._v("36")]),n("br"),n("span",{staticClass:"line-number"},[s._v("37")]),n("br"),n("span",{staticClass:"line-number"},[s._v("38")]),n("br"),n("span",{staticClass:"line-number"},[s._v("39")]),n("br"),n("span",{staticClass:"line-number"},[s._v("40")]),n("br")])]),n("h4",{attrs:{id:"反向代理"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#反向代理"}},[s._v("#")]),s._v(" 反向代理")]),s._v(" "),n("p",[s._v("以下是最简单的反向代理，但由于CA证书缘故，反向代理不支持HTTPS")]),s._v(" "),n("div",{staticClass:"language- line-numbers-mode"},[n("pre",{pre:!0,attrs:{class:"language-text"}},[n("code",[s._v("location / {  \n    proxy_pass http://www.baidu.com;  \n    # 页面资源的存放目录，这里是../html下  \n    #root   html;  \n    #index  index.html index.htm;\n    }\n")])]),s._v(" "),n("div",{staticClass:"line-numbers-wrapper"},[n("span",{staticClass:"line-number"},[s._v("1")]),n("br"),n("span",{staticClass:"line-number"},[s._v("2")]),n("br"),n("span",{staticClass:"line-number"},[s._v("3")]),n("br"),n("span",{staticClass:"line-number"},[s._v("4")]),n("br"),n("span",{staticClass:"line-number"},[s._v("5")]),n("br"),n("span",{staticClass:"line-number"},[s._v("6")]),n("br")])]),n("p",[s._v("多服务器反向代理")]),s._v(" "),n("div",{staticClass:"language-txt line-numbers-mode"},[n("pre",{pre:!0,attrs:{class:"language-txt"}},[n("code",[s._v("http {  \n    # 决定返回文件的类型，例如: jpg, html, css...  \n    # mime.types可以查看conf/mime.types  \n    include       mime.types;  \n    # 如果不包含在mime.types里，则默认为application/octet-stream  \n    default_type  application/octet-stream;  \n  \n    sendfile        on;  \n    # 响应超时范围  \n    keepalive_timeout  65;  \n  \n    upstream httpds {  \n        server 192.168.229.134:80;  \n    }  \n    # 虚拟主机  \n    server {  \n        # 监听端口号  \n        listen       80;  \n        # 服务器IP  \n        server_name  localhost;  \n  \n        location / {  \n            proxy_pass http://httpds;  \n            # 页面资源的存放目录，这里是../html下  \n            #root   html;  \n            #index  index.html index.htm;        }  \n        # 当响应是500 502 503 504时 返回50x.html页面  \n        error_page   500 502 503 504  /50x.html;  \n        # 当用户访问50x.html找不到时，会去../html目录下找  \n        location = /50x.html {  \n            root   html;  \n        }  \n    }  \n}\n")])]),s._v(" "),n("div",{staticClass:"line-numbers-wrapper"},[n("span",{staticClass:"line-number"},[s._v("1")]),n("br"),n("span",{staticClass:"line-number"},[s._v("2")]),n("br"),n("span",{staticClass:"line-number"},[s._v("3")]),n("br"),n("span",{staticClass:"line-number"},[s._v("4")]),n("br"),n("span",{staticClass:"line-number"},[s._v("5")]),n("br"),n("span",{staticClass:"line-number"},[s._v("6")]),n("br"),n("span",{staticClass:"line-number"},[s._v("7")]),n("br"),n("span",{staticClass:"line-number"},[s._v("8")]),n("br"),n("span",{staticClass:"line-number"},[s._v("9")]),n("br"),n("span",{staticClass:"line-number"},[s._v("10")]),n("br"),n("span",{staticClass:"line-number"},[s._v("11")]),n("br"),n("span",{staticClass:"line-number"},[s._v("12")]),n("br"),n("span",{staticClass:"line-number"},[s._v("13")]),n("br"),n("span",{staticClass:"line-number"},[s._v("14")]),n("br"),n("span",{staticClass:"line-number"},[s._v("15")]),n("br"),n("span",{staticClass:"line-number"},[s._v("16")]),n("br"),n("span",{staticClass:"line-number"},[s._v("17")]),n("br"),n("span",{staticClass:"line-number"},[s._v("18")]),n("br"),n("span",{staticClass:"line-number"},[s._v("19")]),n("br"),n("span",{staticClass:"line-number"},[s._v("20")]),n("br"),n("span",{staticClass:"line-number"},[s._v("21")]),n("br"),n("span",{staticClass:"line-number"},[s._v("22")]),n("br"),n("span",{staticClass:"line-number"},[s._v("23")]),n("br"),n("span",{staticClass:"line-number"},[s._v("24")]),n("br"),n("span",{staticClass:"line-number"},[s._v("25")]),n("br"),n("span",{staticClass:"line-number"},[s._v("26")]),n("br"),n("span",{staticClass:"line-number"},[s._v("27")]),n("br"),n("span",{staticClass:"line-number"},[s._v("28")]),n("br"),n("span",{staticClass:"line-number"},[s._v("29")]),n("br"),n("span",{staticClass:"line-number"},[s._v("30")]),n("br"),n("span",{staticClass:"line-number"},[s._v("31")]),n("br"),n("span",{staticClass:"line-number"},[s._v("32")]),n("br"),n("span",{staticClass:"line-number"},[s._v("33")]),n("br"),n("span",{staticClass:"line-number"},[s._v("34")]),n("br")])]),n("h4",{attrs:{id:"负载均衡"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#负载均衡"}},[s._v("#")]),s._v(" 负载均衡")]),s._v(" "),n("h6",{attrs:{id:"轮询权重"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#轮询权重"}},[s._v("#")]),s._v(" 轮询权重")]),s._v(" "),n("p",[s._v("![[Pasted image 20220519185326.png]]\n假设我们有两台后台服务器，这两台服务器的带宽不一，一个是1000M，另一个是100M，我们希望1000M的服务器能负责8次服务，100M的服务器负责2次服务。那么我们该如何设置呢？")]),s._v(" "),n("div",{staticClass:"language-txt line-numbers-mode"},[n("pre",{pre:!0,attrs:{class:"language-txt"}},[n("code",[s._v("upstream httpds {  \n    server 192.168.229.134:80 weight=8;  \n    server 192.168.229.135:80 weight=2;\n}\n")])]),s._v(" "),n("div",{staticClass:"line-numbers-wrapper"},[n("span",{staticClass:"line-number"},[s._v("1")]),n("br"),n("span",{staticClass:"line-number"},[s._v("2")]),n("br"),n("span",{staticClass:"line-number"},[s._v("3")]),n("br"),n("span",{staticClass:"line-number"},[s._v("4")]),n("br")])]),n("h6",{attrs:{id:"down"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#down"}},[s._v("#")]),s._v(" down")]),s._v(" "),n("p",[s._v("那么问题来了，我们给134配置的权重过大导致他超出我们预期的负荷，此时我们希望他能“休息一下”该如何实现呢？")]),s._v(" "),n("div",{staticClass:"language-txt line-numbers-mode"},[n("pre",{pre:!0,attrs:{class:"language-txt"}},[n("code",[s._v("upstream httpds {  \n    server 192.168.229.134:80 weight=8 down;  \n    server 192.168.229.135:80 weight=2;\n}\n")])]),s._v(" "),n("div",{staticClass:"line-numbers-wrapper"},[n("span",{staticClass:"line-number"},[s._v("1")]),n("br"),n("span",{staticClass:"line-number"},[s._v("2")]),n("br"),n("span",{staticClass:"line-number"},[s._v("3")]),n("br"),n("span",{staticClass:"line-number"},[s._v("4")]),n("br")])]),n("h6",{attrs:{id:"backup"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#backup"}},[s._v("#")]),s._v(" backup")]),s._v(" "),n("p",[s._v("你搭建的电商平台在双十一访问量爆炸式增长，你打算设置一个备用服务器在高负载情况下响应请求，该如何做到呢？")]),s._v(" "),n("div",{staticClass:"language-txt line-numbers-mode"},[n("pre",{pre:!0,attrs:{class:"language-txt"}},[n("code",[s._v("upstream httpds {  \n    server 192.168.229.134:80 weight=8 down;  \n    server 192.168.229.135:80 weight=2;\n    server 192.168.229.136:80 backup;\n}\n")])]),s._v(" "),n("div",{staticClass:"line-numbers-wrapper"},[n("span",{staticClass:"line-number"},[s._v("1")]),n("br"),n("span",{staticClass:"line-number"},[s._v("2")]),n("br"),n("span",{staticClass:"line-number"},[s._v("3")]),n("br"),n("span",{staticClass:"line-number"},[s._v("4")]),n("br"),n("span",{staticClass:"line-number"},[s._v("5")]),n("br")])]),n("p",[s._v("被设置backup的服务器只会在其余服务器皆不可用时才能被访问，因此成为了备用服务器。\n但实际场景中，down和backup都不常用，既然你已经知道服务器down了，你可以重启而不是备注一个down标签。")]),s._v(" "),n("p",[s._v("此外，上述weight、down、backup属于轮询策略，无法保持会话。")])])}),[],!1,null,null,null);n.default=e.exports}}]);