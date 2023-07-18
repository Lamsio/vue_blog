(window.webpackJsonp=window.webpackJsonp||[]).push([[116],{445:function(a,s,e){"use strict";e.r(s);var t=e(4),v=Object(t.a)({},(function(){var a=this,s=a._self._c;return s("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[s("h4",{attrs:{id:"简介"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#简介"}},[a._v("#")]),a._v(" 简介")]),a._v(" "),s("p",[a._v("在前几章中，我们通过动态路由协议实现了数据包的收发路由设定，但通常这只用于特定区域网络环境下，更多情况下，我们的电脑都是只有内网地址，此时我们想与外网建立连接，总不能在目标设备上也设定回传的路由吧。")]),a._v(" "),s("p",[s("img",{attrs:{src:"/more/Pasted%20image%2020221220150141.png",alt:"avatar"}})]),a._v(" "),s("p",[a._v("如上图所示，PC1想和AR2建立连接，很明显PC1是内网IP，因此根本就不可能在AR2上设定回传路由，此时我们就需要用到NAT进行网络地址转换了。")]),a._v(" "),s("p",[a._v("该功能能将源地址为内网IP发出的包改成发出口，即公网IP作为源地址，如上图，当包从"),s("code",[a._v("12.0.0.1")]),a._v("传出时，源地址将不再是"),s("code",[a._v("192.168.1.1")]),a._v("。")]),a._v(" "),s("h4",{attrs:{id:"配置"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#配置"}},[a._v("#")]),a._v(" 配置")]),a._v(" "),s("p",[a._v("我们可以用先前学到的ACL为NAT确立哪些包需要改源IP，哪些包不需要改。")]),a._v(" "),s("p",[a._v("首先我们先配置ACL")]),a._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[a._v("[Huawei] acl 2000\n[Huawei-acl-2000] rule 5 permit source 192.168.1.0 0.0.0.255\n")])]),a._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[a._v("1")]),s("br"),s("span",{staticClass:"line-number"},[a._v("2")]),s("br")])]),s("p",[a._v("然后进入指定接口配置NAT")]),a._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[a._v("[Huawei] int g 0/0/2\n[Huawei] nat outbound 2000\n")])]),a._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[a._v("1")]),s("br"),s("span",{staticClass:"line-number"},[a._v("2")]),s("br")])]),s("p",[a._v("这样，ACL2000的规则就被配置到接口0/0/2了，当有数据传出时，就会运行检测，如果符合ACL配置，则将源地址改为出口地址。")]),a._v(" "),s("h4",{attrs:{id:"地址池"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#地址池"}},[a._v("#")]),a._v(" 地址池")]),a._v(" "),s("p",[a._v("个别大公司可能网络规模是十分巨大，这意味着个别网段根本就不够公司使用，因此他们可能会将某个区间的网段划分为地址池，这样可以避免手动地一个个配置。")]),a._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[a._v("[Huawei] nat address-group 1 12.1.1.10 12.1.1.20\n[Huawei] int g 0/0/2\n[Huawei-GE0/0/2] nat outbound 2000 address-group 1\n")])]),a._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[a._v("1")]),s("br"),s("span",{staticClass:"line-number"},[a._v("2")]),s("br"),s("span",{staticClass:"line-number"},[a._v("3")]),s("br")])]),s("p",[a._v("上面的指令是使从"),s("code",[a._v("12.0.0.1")]),a._v("接口发出的包拥有"),s("code",[a._v("12.1.1.10~12.1.1.20")]),a._v("连续的可规划源IP，这意味着PC1在时刻A发出的包可能是"),s("code",[a._v("12.1.1.10")]),a._v("，在时刻B时发出的包则变为"),s("code",[a._v("12.1.1.17")]),a._v("，分配是由设备自行处理。")]),a._v(" "),s("h4",{attrs:{id:"ip购买后的配置"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#ip购买后的配置"}},[a._v("#")]),a._v(" IP购买后的配置")]),a._v(" "),s("p",[a._v("公司向ISP购买了IP地址，公司希望将公网IP地址绑定给公司内部的服务器用于对外提供访问服务，按先前的知识我们知道，通过NAT我们能将内网IP的数据包发送给外网，但很明显有个弊端，那就是外网无法访问内网所提供的服务。")]),a._v(" "),s("p",[a._v("因此我们需要在网关设备上配置一个NAT映射，外网将访问特定的公网IP，当数据包传到设备网关时，设备会将外网的请求转发给内网。")]),a._v(" "),s("p",[a._v("假设我们现在有个"),s("code",[a._v("100.1.1.1/32")]),a._v("的公网IP\n我们在网关设备，即AR1中的GE0/0/1接口配置"),s("code",[a._v("nat server global 100.1.1.1 inside 192.168.1.1")]),a._v("，然后在ISP的路由表中添加一条静态路由"),s("code",[a._v("ip route-static 100.1.1.1 32 12.0.0.1")]),a._v("这样，当外部网络访问"),s("code",[a._v("100.1.1.1")]),a._v("时，ISP会将其导向"),s("code",[a._v("12.0.0.1")]),a._v("，而AR1的GE0/0/1则会通过设置的NAT映射将"),s("code",[a._v("100.1.1.1")]),a._v("的流量转发给"),s("code",[a._v("192.168.1.1")])]),a._v(" "),s("p",[a._v("但这种方式的问题是会将内部设备完完全全地暴露在公网环境下，大大增加了被攻击的风险。因此我们更推荐使用"),s("strong",[a._v("NAT端口转发")])]),a._v(" "),s("p",[a._v("我们可以根据IP层协议、端口设定转发")]),a._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[a._v("[Huawei] int g 0/0/1\n[Huawei] nat server protocol tcp global 100.1.1.1 80 inside 192.168.1.1 80\n")])]),a._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[a._v("1")]),s("br"),s("span",{staticClass:"line-number"},[a._v("2")]),s("br")])]),s("p",[a._v("这样设定完成后，对外仅转发tcp协议80端口的数据，同时，IP的复用率大大提高，由于我们只用了一个端口，这意味着相同的IP我们还能设置不同的转发，例如22端口的数据转发给PC2。"),s("strong",[a._v("切记，一定要配置在数据流入口，即上图的12.0.0.1的那个口")])])])}),[],!1,null,null,null);s.default=v.exports}}]);