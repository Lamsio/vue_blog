(window.webpackJsonp=window.webpackJsonp||[]).push([[55],{384:function(s,n,a){"use strict";a.r(n);var t=a(4),r=Object(t.a)({},(function(){var s=this,n=s._self._c;return n("ContentSlotsDistributor",{attrs:{"slot-key":s.$parent.slotKey}},[n("h2",{attrs:{id:"跨域问题"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#跨域问题"}},[s._v("#")]),s._v(" 跨域问题")]),s._v(" "),n("h4",{attrs:{id:"同源策略"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#同源策略"}},[s._v("#")]),s._v(" 同源策略")]),s._v(" "),n("p",[s._v("一种约定，是浏览器最核心也是最基本的安全功能，如果浏览器少了同源策略，则浏览器的正常功能也会受影响。")]),s._v(" "),n("p",[s._v("同源：协议、域名（IP）、端口相同即同源")]),s._v(" "),n("div",{staticClass:"language-txt line-numbers-mode"},[n("pre",{pre:!0,attrs:{class:"language-txt"}},[n("code",[s._v("http://192.168.1.1/usr/1\nhttps://192.168.1.1/usr/1\n协议不同，因此不是\n\nhttp://192.168.1.1/usr/1\nhttp://192.168.1.2/usr/1\nip不同，因此不是\n\nhttp://192.168.1.1/usr/1\nhttp://192.168.1.1:8080/usr/1\n端口不同，因此不是\n\nhttp://www.nginx.com/usr/1\nhttp://www.nginx.org/usr/1\n域名不同，因此不是\n\nhttp://www.nginx.com/usr/1\nhttp://www.nginx.com:80/usr/1\n满足，因此是\n")])]),s._v(" "),n("div",{staticClass:"line-numbers-wrapper"},[n("span",{staticClass:"line-number"},[s._v("1")]),n("br"),n("span",{staticClass:"line-number"},[s._v("2")]),n("br"),n("span",{staticClass:"line-number"},[s._v("3")]),n("br"),n("span",{staticClass:"line-number"},[s._v("4")]),n("br"),n("span",{staticClass:"line-number"},[s._v("5")]),n("br"),n("span",{staticClass:"line-number"},[s._v("6")]),n("br"),n("span",{staticClass:"line-number"},[s._v("7")]),n("br"),n("span",{staticClass:"line-number"},[s._v("8")]),n("br"),n("span",{staticClass:"line-number"},[s._v("9")]),n("br"),n("span",{staticClass:"line-number"},[s._v("10")]),n("br"),n("span",{staticClass:"line-number"},[s._v("11")]),n("br"),n("span",{staticClass:"line-number"},[s._v("12")]),n("br"),n("span",{staticClass:"line-number"},[s._v("13")]),n("br"),n("span",{staticClass:"line-number"},[s._v("14")]),n("br"),n("span",{staticClass:"line-number"},[s._v("15")]),n("br"),n("span",{staticClass:"line-number"},[s._v("16")]),n("br"),n("span",{staticClass:"line-number"},[s._v("17")]),n("br"),n("span",{staticClass:"line-number"},[s._v("18")]),n("br"),n("span",{staticClass:"line-number"},[s._v("19")]),n("br")])]),n("h4",{attrs:{id:"跨域"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#跨域"}},[s._v("#")]),s._v(" 跨域")]),s._v(" "),n("p",[s._v("当有两台服务器分别为A、B，如果从服务器A页面发送异步请求到服务器B获取数据，如果服务器A和服务器B不满足同源政策就会发生跨域问题。")]),s._v(" "),n("h6",{attrs:{id:"解决"}},[n("a",{staticClass:"header-anchor",attrs:{href:"#解决"}},[s._v("#")]),s._v(" 解决")]),s._v(" "),n("div",{staticClass:"language- line-numbers-mode"},[n("pre",{pre:!0,attrs:{class:"language-text"}},[n("code",[s._v("## 在被请求的服务器下加add_header\nlocation / {\n\tadd_header Access-Control-Allow-Origin http://192.168.229.133;\n\tadd_header Access-Control-Allow-Methods GET,POST;\n\t...\n}\n")])]),s._v(" "),n("div",{staticClass:"line-numbers-wrapper"},[n("span",{staticClass:"line-number"},[s._v("1")]),n("br"),n("span",{staticClass:"line-number"},[s._v("2")]),n("br"),n("span",{staticClass:"line-number"},[s._v("3")]),n("br"),n("span",{staticClass:"line-number"},[s._v("4")]),n("br"),n("span",{staticClass:"line-number"},[s._v("5")]),n("br"),n("span",{staticClass:"line-number"},[s._v("6")]),n("br")])])])}),[],!1,null,null,null);n.default=r.exports}}]);