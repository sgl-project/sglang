# A basic mkdocstrings handler for {fmt}.
# Copyright (c) 2012 - present, Victor Zverovich
# https://github.com/fmtlib/fmt/blob/master/LICENSE

import os
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from subprocess import PIPE, STDOUT, CalledProcessError, Popen
from typing import Any, List, Mapping, Optional

from mkdocstrings.handlers.base import BaseHandler


class Definition:
    """A definition extracted by Doxygen."""

    def __init__(self, name: str, kind: Optional[str] = None,
                 node: Optional[ElementTree.Element] = None,
                 is_member: bool = False):
        self.name = name
        self.kind = kind if kind is not None else node.get('kind')
        self.desc = None
        self.id = name if not is_member else None
        self.members = None
        self.params = None
        self.template_params = None
        self.trailing_return_type = None
        self.type = None


# A map from Doxygen to HTML tags.
tag_map = {
    'bold': 'b',
    'emphasis': 'em',
    'computeroutput': 'code',
    'para': 'p',
    'programlisting': 'pre',
    'verbatim': 'pre'
}

# A map from Doxygen tags to text.
tag_text_map = {
    'codeline': '',
    'highlight': '',
    'sp': ' '
}


def escape_html(s: str) -> str:
    return s.replace("<", "&lt;")


def doxyxml2html(nodes: List[ElementTree.Element]):
    out = ''
    for n in nodes:
        tag = tag_map.get(n.tag)
        if not tag:
            out += tag_text_map[n.tag]
        out += '<' + tag + '>' if tag else ''
        out += '<code class="language-cpp">' if tag == 'pre' else ''
        if n.text:
            out += escape_html(n.text)
        out += doxyxml2html(list(n))
        out += '</code>' if tag == 'pre' else ''
        out += '</' + tag + '>' if tag else ''
        if n.tail:
            out += n.tail
    return out


def convert_template_params(node: ElementTree.Element) -> Optional[List[Definition]]:
    template_param_list = node.find('templateparamlist')
    if template_param_list is None:
        return None
    params = []
    for param_node in template_param_list.findall('param'):
        name = param_node.find('declname')
        param = Definition(name.text if name is not None else '', 'param')
        param.type = param_node.find('type').text
        params.append(param)
    return params


def get_description(node: ElementTree.Element) -> List[ElementTree.Element]:
    return node.findall('briefdescription/para') + \
        node.findall('detaileddescription/para')


def normalize_type(type_: str) -> str:
    type_ = type_.replace('< ', '<').replace(' >', '>')
    return type_.replace(' &', '&').replace(' *', '*')


def convert_type(type_: ElementTree.Element) -> Optional[str]:
    if type_ is None:
        return None
    result = type_.text if type_.text else ''
    for ref in type_:
        result += ref.text
        if ref.tail:
            result += ref.tail
    result += type_.tail.strip()
    return normalize_type(result)


def convert_params(func: ElementTree.Element) -> List[Definition]:
    params = []
    for p in func.findall('param'):
        d = Definition(p.find('declname').text, 'param')
        d.type = convert_type(p.find('type'))
        params.append(d)
    return params


def convert_return_type(d: Definition, node: ElementTree.Element) -> None:
    d.trailing_return_type = None
    if d.type == 'auto' or d.type == 'constexpr auto':
        parts = node.find('argsstring').text.split(' -> ')
        if len(parts) > 1:
            d.trailing_return_type = normalize_type(parts[1])


def render_param(param: Definition) -> str:
    return param.type + (f'&nbsp;{param.name}' if len(param.name) > 0 else '')


def render_decl(d: Definition) -> str:
    text = ''
    if d.id is not None:
        text += f'<a id="{d.id}">\n'
    text += '<pre><code class="language-cpp decl">'

    text += '<div>'
    if d.template_params is not None:
        text += 'template &lt;'
        text += ', '.join([render_param(p) for p in d.template_params])
        text += '&gt;\n'
    text += '</div>'

    text += '<div>'
    end = ';'
    if d.kind == 'function' or d.kind == 'variable':
        text += d.type + ' ' if len(d.type) > 0 else ''
    elif d.kind == 'typedef':
        text += 'using '
    elif d.kind == 'define':
        end = ''
    else:
        text += d.kind + ' '
    text += d.name

    if d.params is not None:
        params = ', '.join([
            (p.type + ' ' if p.type else '') + p.name for p in d.params])
        text += '(' + escape_html(params) + ')'
        if d.trailing_return_type:
            text += ' -&NoBreak;>&nbsp;' + escape_html(d.trailing_return_type)
    elif d.kind == 'typedef':
        text += ' = ' + escape_html(d.type)

    text += end
    text += '</div>'
    text += '</code></pre>\n'
    if d.id is not None:
        text += f'</a>\n'
    return text


class CxxHandler(BaseHandler):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(handler='cxx', **kwargs)

        headers = [
            'args.h', 'base.h', 'chrono.h', 'color.h', 'compile.h', 'format.h',
            'os.h', 'ostream.h', 'printf.h', 'ranges.h', 'std.h', 'xchar.h'
        ]

        # Run doxygen.
        cmd = ['doxygen', '-']
        support_dir = Path(__file__).parents[3]
        top_dir = os.path.dirname(support_dir)
        include_dir = os.path.join(top_dir, 'include', 'fmt')
        self._ns2doxyxml = {}
        build_dir = os.path.join(top_dir, 'build')
        os.makedirs(build_dir, exist_ok=True)
        self._doxyxml_dir = os.path.join(build_dir, 'doxyxml')
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
        _, _ = p.communicate(input=r'''
            PROJECT_NAME     = fmt
            GENERATE_XML     = YES
            GENERATE_LATEX   = NO
            GENERATE_HTML    = NO
            INPUT            = {0}
            XML_OUTPUT       = {1}
            QUIET            = YES
            AUTOLINK_SUPPORT = NO
            MACRO_EXPANSION  = YES
            PREDEFINED       = _WIN32=1 \
                               __linux__=1 \
                               FMT_ENABLE_IF(...)= \
                               FMT_USE_USER_LITERALS=1 \
                               FMT_USE_ALIAS_TEMPLATES=1 \
                               FMT_USE_NONTYPE_TEMPLATE_ARGS=1 \
                               FMT_API= \
                               "FMT_BEGIN_NAMESPACE=namespace fmt {{" \
                               "FMT_END_NAMESPACE=}}" \
                               "FMT_DOC=1"
            '''.format(
                ' '.join([os.path.join(include_dir, h) for h in headers]),
                self._doxyxml_dir).encode('utf-8'))
        if p.returncode != 0:
            raise CalledProcessError(p.returncode, cmd)

        # Merge all file-level XMLs into one to simplify search.
        self._file_doxyxml = None
        for h in headers:
            filename = h.replace(".h", "_8h.xml")
            with open(os.path.join(self._doxyxml_dir, filename)) as f:
                doxyxml = ElementTree.parse(f)
                if self._file_doxyxml is None:
                    self._file_doxyxml = doxyxml
                    continue
                root = self._file_doxyxml.getroot()
                for node in doxyxml.getroot():
                    root.append(node)

    def collect_compound(self, identifier: str,
                         cls: List[ElementTree.Element]) -> Definition:
        """Collect a compound definition such as a struct."""
        path = os.path.join(self._doxyxml_dir, cls[0].get('refid') + '.xml')
        with open(path) as f:
            xml = ElementTree.parse(f)
            node = xml.find('compounddef')
            d = Definition(identifier, node=node)
            d.template_params = convert_template_params(node)
            d.desc = get_description(node)
            d.members = []
            for m in \
                    node.findall('sectiondef[@kind="public-attrib"]/memberdef') + \
                    node.findall('sectiondef[@kind="public-func"]/memberdef'):
                name = m.find('name').text
                # Doxygen incorrectly classifies members of private unnamed unions as
                # public members of the containing class.
                if name.endswith('_'):
                    continue
                desc = get_description(m)
                if len(desc) == 0:
                    continue
                kind = m.get('kind')
                member = Definition(name if name else '', kind=kind, is_member=True)
                type_text = m.find('type').text
                member.type = type_text if type_text else ''
                if kind == 'function':
                    member.params = convert_params(m)
                    convert_return_type(member, m)
                member.template_params = None
                member.desc = desc
                d.members.append(member)
            return d

    def collect(self, identifier: str, _config: Mapping[str, Any]) -> Definition:
        qual_name = 'fmt::' + identifier

        param_str = None
        paren = qual_name.find('(')
        if paren > 0:
            qual_name, param_str = qual_name[:paren], qual_name[paren + 1:-1]

        colons = qual_name.rfind('::')
        namespace, name = qual_name[:colons], qual_name[colons + 2:]

        # Load XML.
        doxyxml = self._ns2doxyxml.get(namespace)
        if doxyxml is None:
            path = f'namespace{namespace.replace("::", "_1_1")}.xml'
            with open(os.path.join(self._doxyxml_dir, path)) as f:
                doxyxml = ElementTree.parse(f)
                self._ns2doxyxml[namespace] = doxyxml

        nodes = doxyxml.findall(
            f"compounddef/sectiondef/memberdef/name[.='{name}']/..")
        if len(nodes) == 0:
            nodes = self._file_doxyxml.findall(
                f"compounddef/sectiondef/memberdef/name[.='{name}']/..")
        candidates = []
        for node in nodes:
            # Process a function or a typedef.
            params = None
            d = Definition(name, node=node)
            if d.kind == 'function':
                params = convert_params(node)
                node_param_str = ', '.join([p.type for p in params])
                if param_str and param_str != node_param_str:
                    candidates.append(f'{name}({node_param_str})')
                    continue
            elif d.kind == 'define':
                params = []
                for p in node.findall('param'):
                    param = Definition(p.find('defname').text, kind='param')
                    param.type = None
                    params.append(param)
            d.type = convert_type(node.find('type'))
            d.template_params = convert_template_params(node)
            d.params = params
            convert_return_type(d, node)
            d.desc = get_description(node)
            return d

        cls = doxyxml.findall(f"compounddef/innerclass[.='{qual_name}']")
        if not cls:
            raise Exception(f'Cannot find {identifier}. Candidates: {candidates}')
        return self.collect_compound(identifier, cls)

    def render(self, d: Definition, config: dict) -> str:
        if d.id is not None:
            self.do_heading('', 0, id=d.id)
        text = '<div class="docblock">\n'
        text += render_decl(d)
        text += '<div class="docblock-desc">\n'
        text += doxyxml2html(d.desc)
        if d.members is not None:
            for m in d.members:
                text += self.render(m, config)
        text += '</div>\n'
        text += '</div>\n'
        return text


def get_handler(theme: str, custom_templates: Optional[str] = None,
                **_config: Any) -> CxxHandler:
    """Return an instance of `CxxHandler`.

    Arguments:
        theme: The theme to use when rendering contents.
        custom_templates: Directory containing custom templates.
        **_config: Configuration passed to the handler.
    """
    return CxxHandler(theme=theme, custom_templates=custom_templates)
