(window.webpackJsonp=window.webpackJsonp||[]).push([[134],{462:function(a,t,r){"use strict";r.r(t);var s=r(4),e=Object(s.a)({},(function(){var a=this,t=a._self._c;return t("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220529144444.png",alt:"avatar"}}),a._v(" "),t("img",{attrs:{src:"/more/Pasted%20image%2020220529144808.png",alt:"avatar"}}),a._v(" "),t("img",{attrs:{src:"/more/Pasted%20image%2020220529144908.png",alt:"avatar"}}),a._v(" "),t("img",{attrs:{src:"/more/Pasted%20image%2020220529145142.png",alt:"avatar"}}),a._v(" "),t("img",{attrs:{src:"/more/Pasted%20image%2020220529151013.png",alt:"avatar"}})]),a._v(" "),t("p",[a._v("一开始Client需要发送一个报文给目标服务器，以上图为例seq为100.\n当服务器收到后，需要作出回应，返回了ack101代表收到了seq为100的请求。\n同时将这条信息告知给Client，双方根据自己的seq和ack确立与谁进行通信\n"),t("img",{attrs:{src:"/more/Pasted%20image%2020220529151543.png",alt:"avatar"}})]),a._v(" "),t("h4",{attrs:{id:"窗口机制"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#窗口机制"}},[a._v("#")]),a._v(" 窗口机制")]),a._v(" "),t("p",[a._v("有时，每次只发送一条信息属实浪费时间，因此我们可以一次性发送多条信息，在发送信息时需要告知size，如下图，A发送的size为3意味着一次发了三条信息，但B回答的是2，意味着B的缓存只能接收2条，也意味着漏了一条")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220529151842.png",alt:"avatar"}})]),a._v(" "),t("h4",{attrs:{id:"网络层"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#网络层"}},[a._v("#")]),a._v(" 网络层")]),a._v(" "),t("p",[t("img",{attrs:{src:"/more/Pasted%20image%2020220529152312.png",alt:"avatar"}}),a._v(" "),t("img",{attrs:{src:"/more/Pasted%20image%2020220529152427.png",alt:"avatar"}}),a._v(" "),t("img",{attrs:{src:"/more/Pasted%20image%2020220529152459.png",alt:"avatar"}}),a._v(" "),t("img",{attrs:{src:"/more/Pasted%20image%2020220529152616.png",alt:"avatar"}})]),a._v(" "),t("p",[a._v("当局域网A的用户打算发信息给局域网B的用户时。\nMAC地址仅作用于LAN，IP地址则作用于WAN")]),a._v(" "),t("p",[a._v("那么问题来了，主机A和C在局域网A中，那么他俩如何得知对方MAC地址的呢？\n每台主机会有一个ARP映射表，包含了主机IP以及MAC地址。主机A通过广播形式请求IP B的MAC地址，该广播请求会被局域网内所有主机监听到，但只有B会回复。Windows ARP查询: arp -a\n"),t("img",{attrs:{src:"/more/Pasted%20image%2020220529153939.png",alt:"avatar"}})]),a._v(" "),t("h4",{attrs:{id:"vlsm"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#vlsm"}},[a._v("#")]),a._v(" VLSM")])])}),[],!1,null,null,null);t.default=e.exports}}]);