(window.webpackJsonp=window.webpackJsonp||[]).push([[110],{438:function(a,t,e){"use strict";e.r(t);var v=e(4),_=Object(v.a)({},(function(){var a=this,t=a._self._c;return t("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[t("h2",{attrs:{id:"聚合与继承"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#聚合与继承"}},[a._v("#")]),a._v(" 聚合与继承")]),a._v(" "),t("h4",{attrs:{id:"聚合"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#聚合"}},[a._v("#")]),a._v(" 聚合")]),a._v(" "),t("p",[a._v("![[Pasted image 20220617152042.png]]")]),a._v(" "),t("p",[a._v("已知"),t("code",[a._v("crm")]),a._v("、"),t("code",[a._v("order")]),a._v("、"),t("code",[a._v("member")]),a._v("需要依赖"),t("code",[a._v("pojo")]),a._v("，倘若"),t("code",[a._v("pojo")]),a._v("更新后导致这三个项目无法使用怎么办？")]),a._v(" "),t("p",[a._v("![[Pasted image 20220617152251.png]]")]),a._v(" "),t("h6",{attrs:{id:"流程"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#流程"}},[a._v("#")]),a._v(" 流程")]),a._v(" "),t("p",[a._v("![[Pasted image 20220617162959.png]]\n![[Pasted image 20220617163012.png]]")]),a._v(" "),t("h4",{attrs:{id:"继承"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#继承"}},[a._v("#")]),a._v(" 继承")]),a._v(" "),t("p",[a._v("![[Pasted image 20220617163124.png]]\n上图中，你能看到三个子模块各自的依赖项都有相同和相似处，思考：")]),a._v(" "),t("ol",[t("li",[a._v("完全相同能否简化？")]),a._v(" "),t("li",[a._v("有一部分配置一样能否简化？")])]),a._v(" "),t("h6",{attrs:{id:"子工程定义继承关系"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#子工程定义继承关系"}},[a._v("#")]),a._v(" 子工程定义继承关系")]),a._v(" "),t("p",[a._v("子工程需要在自己的pom.xml文件中定义如下内容：\n![[Pasted image 20220617163802.png]]")]),a._v(" "),t("p",[a._v("然后父工程只需要配置子工程共用的依赖库即可，往后子工程自身的pom.xml只需要写自身使用的依赖库即可，一些通用的依赖库可以写在父工程pom.xml文件中，从而减少开发中的重复依赖")]),a._v(" "),t("h6",{attrs:{id:"依赖管理"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#依赖管理"}},[a._v("#")]),a._v(" 依赖管理")]),a._v(" "),t("p",[a._v("![[Pasted image 20220617164506.png]]\n上述配置是在父工程pom.xml中的，用于对子工程可选依赖进行管理。\n意思是某些子工程可能需要用到junit库，当他们在自己的pom.xml配置junit依赖时，会根据父工程配置的依赖管理进行处理，也就是说，即使子工程打算依赖junit库时，不需要配置版本，只需要配置其"),t("code",[a._v("artifactID")]),a._v("和"),t("code",[a._v("groupID")]),a._v("即可，这样就会按照父工程中的依赖管理进行处理了。")]),a._v(" "),t("h4",{attrs:{id:"区别"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#区别"}},[a._v("#")]),a._v(" 区别")]),a._v(" "),t("p",[a._v("![[Pasted image 20220617165038.png]]")])])}),[],!1,null,null,null);t.default=_.exports}}]);