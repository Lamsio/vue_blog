(window.webpackJsonp=window.webpackJsonp||[]).push([[57],{386:function(t,s,a){"use strict";a.r(s);var e=a(4),n=Object(e.a)({},(function(){var t=this,s=t._self._c;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("h2",{attrs:{id:"rewrite功能"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#rewrite功能"}},[t._v("#")]),t._v(" Rewrite功能")]),t._v(" "),s("p",[t._v("这是Nginx服务器提供的一个重要的基本功能，是WEB服务器产品中几乎必备的功能。主要作用是用来实现URL重写。\n例如你输入 www.360buy.com 会自动跳转到 www.jd.com")]),t._v(" "),s("h4",{attrs:{id:"set指令"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#set指令"}},[t._v("#")]),t._v(" set指令")]),t._v(" "),s("table",[s("thead",[s("tr",[s("th",[t._v("'描述'")]),t._v(" "),s("th",[t._v("'指令'")])])]),t._v(" "),s("tbody",[s("tr",[s("td",[t._v("语法")]),t._v(" "),s("td",[t._v("set $variable value;")])]),t._v(" "),s("tr",[s("td",[t._v("默认值")]),t._v(" "),s("td",[t._v("-")])]),t._v(" "),s("tr",[s("td",[t._v("位置")]),t._v(" "),s("td",[t._v("server、location、if")])])])]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("server {\n\tlisten 8080;\n\tserver_name localhost;\n\n\tlocation /server {\n\t\tset $name Tommy;\n\t\tset $age 18;\n\t\treturn 200 $name,$age;\n\t}\n}\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br"),s("span",{staticClass:"line-number"},[t._v("9")]),s("br"),s("span",{staticClass:"line-number"},[t._v("10")]),s("br")])]),s("p",[t._v("由于Nginx中有全局变量，因此确保不要命名冲突。")]),t._v(" "),s("h4",{attrs:{id:"rewrite常用全局变量"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#rewrite常用全局变量"}},[t._v("#")]),t._v(" Rewrite常用全局变量")]),t._v(" "),s("table",[s("thead",[s("tr",[s("th",[t._v("变量")]),t._v(" "),s("th",[t._v("说明")])])]),t._v(" "),s("tbody",[s("tr",[s("td",[t._v("$age")]),t._v(" "),s("td",[t._v("存放了URL中的请求参数")])]),t._v(" "),s("tr",[s("td",[t._v("$http_user_agent")]),t._v(" "),s("td",[t._v("变量存储的是用户访问服务的代理信息，例如访问的浏览器信息")])]),t._v(" "),s("tr",[s("td",[t._v("$host")]),t._v(" "),s("td",[t._v("服务器的server_name")])]),t._v(" "),s("tr",[s("td",[t._v("$document_uri")]),t._v(" "),s("td",[t._v("存储了当前访问地址的URI")])])])]),t._v(" "),s("p",[t._v("其余变量查看: https://www.javatpoint.com/nginx-variables")]),t._v(" "),s("h6",{attrs:{id:"这些变量能做什么"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#这些变量能做什么"}},[t._v("#")]),t._v(" 这些变量能做什么？")]),t._v(" "),s("ol",[s("li",[t._v("设置管理日志，记录访问信息")])]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("http{\n\t...\n\tlog_format main '$remote_addr - $request - $status - $request_uri - $http_user_agent';\n\n\n\tserver{\n\t\t...\n\t\tlocation /server {\n\t\t\taccess_log logs/access.log main;\n\t\t\t...\n\t\t}\n\t}\n}\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br"),s("span",{staticClass:"line-number"},[t._v("9")]),s("br"),s("span",{staticClass:"line-number"},[t._v("10")]),s("br"),s("span",{staticClass:"line-number"},[t._v("11")]),s("br"),s("span",{staticClass:"line-number"},[t._v("12")]),s("br"),s("span",{staticClass:"line-number"},[t._v("13")]),s("br")])]),s("h4",{attrs:{id:"if指令"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#if指令"}},[t._v("#")]),t._v(" if指令")]),t._v(" "),s("table",[s("thead",[s("tr",[s("th",[t._v("''")]),t._v(" "),s("th",[t._v("''")])])]),t._v(" "),s("tbody",[s("tr",[s("td",[t._v("语法")]),t._v(" "),s("td",[t._v("if (condition){...}")])]),t._v(" "),s("tr",[s("td",[t._v("默认值")]),t._v(" "),s("td",[t._v("-")])]),t._v(" "),s("tr",[s("td",[t._v("位置")]),t._v(" "),s("td",[t._v("server、location")])])])]),t._v(" "),s("p",[t._v("两种基本用法")]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("if ($params){\n\t...\n}\n\nif ($request_method = POST){\n\treturn 405;\n}\n\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br")])]),s("p",[t._v('正则表达式匹配\n"~"代表匹配正则表达式过程中区分大小写\n"~*"代表匹配正则表达式过程中不区分大小写\n"!~"和"!~*"刚好和上面取相反之，如果匹配上返回false反之true')]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("if ($http_user_agent ~ MSIE){\n\t# $http_user_agent的值中是否包含MSIE字符串，如果包含返回True\n}\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br")])]),s("p",[t._v('判断文件是否存在\n"-f"和"!-f" 文件\n"-d"和"!-d" 目录\n"-e"和"!-e" 文件/目录\n"-x"和"!-x" 判断文件是否可执行')]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("if (-f $request_filename){\n\t#判断请求的文件是否存在\n}\n\nif (!-f $request_filename){\n\t#判断请求文件是否不存在\n}\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br")])]),s("h4",{attrs:{id:"rewrite指令"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#rewrite指令"}},[t._v("#")]),t._v(" rewrite指令")]),t._v(" "),s("p",[t._v("该指令通过正则表达式的使用来改变URI，可以同时存在一个或者多个指令，按照顺序依次对URL进行匹配和处理。")]),t._v(" "),s("table",[s("thead",[s("tr",[s("th",[t._v("''")]),t._v(" "),s("th",[t._v("''")])])]),t._v(" "),s("tbody",[s("tr",[s("td",[t._v("语法")]),t._v(" "),s("td",[t._v("rewrite regex replacement [flag]")])]),t._v(" "),s("tr",[s("td",[t._v("默认值")]),t._v(" "),s("td",[t._v("-")])]),t._v(" "),s("tr",[s("td",[t._v("位置")]),t._v(" "),s("td",[t._v("server、location、if")])])])]),t._v(" "),s("div",{staticClass:"language-txt line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-txt"}},[s("code",[t._v("location /rewrite {\n\trewrite ^/rewrite/url\\w*$ https://www.baidu.com\n}\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br")])]),s("p",[t._v("个人感觉这功能类似于其他web框架中的redirect+正则匹配的结合版。")]),t._v(" "),s("p",[t._v("可用参数:")]),t._v(" "),s("ol",[s("li",[t._v("last - 将匹配的结果值作为新的正则验证值重新进入路由匹配。")]),t._v(" "),s("li",[t._v("break - 将将匹配的结果值作为新的正则验证值重新进入"),s("strong",[t._v("当前")]),t._v("的路由匹配")]),t._v(" "),s("li",[t._v("redirect - 临时重定向")]),t._v(" "),s("li",[t._v("permanent - 永久重定向")])]),t._v(" "),s("h2",{attrs:{id:"案例"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#案例"}},[t._v("#")]),t._v(" 案例")]),t._v(" "),s("h4",{attrs:{id:"多域名重定向"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#多域名重定向"}},[t._v("#")]),t._v(" 多域名重定向")]),t._v(" "),s("p",[t._v("现在我有三个域名")]),t._v(" "),s("ol",[s("li",[t._v("主域名 www.tommy.com")]),t._v(" "),s("li",[t._v("副域名 www.cat.com")]),t._v(" "),s("li",[t._v("副域名 www.dog.com")])]),t._v(" "),s("p",[t._v("我希望访问两个副域名时会自动跳转至主域名该如何实现？")]),t._v(" "),s("div",{staticClass:"language-txt line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-txt"}},[s("code",[t._v("server{\n    listen 80;\n    server_name www.cat.com www.dog.com;\n    #rewrite ^/ http://www.tommy.com;\n    rewrite ^(.*) http://www.tommy.com$1;\n}\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br")])]),s("h4",{attrs:{id:"域名镜像"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#域名镜像"}},[t._v("#")]),t._v(" 域名镜像")]),t._v(" "),s("p",[t._v("上面的案例中，我们实现了域名重定向，但无论如何访问cat和dog域名，都会直接跳转至tommy，此时我们希望能够在这两个副域名下的某个子目录资源做镜像，该如何实现呢？")]),t._v(" "),s("div",{staticClass:"language-txt line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-txt"}},[s("code",[t._v('server {\n    listen 80;\n    server_name www.cat.com www.dog.com;\n    #rewrite ^/ http://www.tommy.com;\n    #rewrite ^(.*) http://www.tommy.com$1;\n    location /user {\n        default_type text/plain;\n        return 200 "chengogn";\n    }\n    location / {\n        rewrite ^(.*) http://www.tommy.com$1;\n    }\n}\n')])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br"),s("span",{staticClass:"line-number"},[t._v("9")]),s("br"),s("span",{staticClass:"line-number"},[t._v("10")]),s("br"),s("span",{staticClass:"line-number"},[t._v("11")]),s("br"),s("span",{staticClass:"line-number"},[t._v("12")]),s("br"),s("span",{staticClass:"line-number"},[t._v("13")]),s("br")])]),s("p",[t._v("如你所见，其实就是把特定目录独立出来而已，其余的依旧是和上面一样。")])])}),[],!1,null,null,null);s.default=n.exports}}]);