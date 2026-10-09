"""Edition-neutral index selection; explicit preferences never silently fall back."""
import json
import os
import re
from pathlib import Path


def normalize_index(value, abspath=None):
    text = str(value or '').strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in '\"\'':
        text = text[1:-1].strip()
    if not text:
        return None
    text = os.path.expandvars(os.path.expanduser(text))
    return Path(abspath(text) if abspath else text)


def steam_libraries(steam):
    """Read additional Steam libraries, including libraries on other drives."""
    steam = Path(steam)
    result = [steam]
    for file in (steam/'steamapps/libraryfolders.vdf', steam/'config/libraryfolders.vdf'):
        try:
            source = file.read_text(encoding='utf-8-sig')
        except (OSError, UnicodeError):
            continue
        for value in re.findall(r'"path"\s*"((?:\\.|[^"\\])*)"', source, re.I):
            result.append(Path(value.replace('\\\\', '\\').replace('\\"', '"')))
    return result


def installed_indexes():
    """Bounded registry/manifest lookup, never a recursive drive scan."""
    candidates = []
    steam_roots = []
    for env in ('ProgramFiles(x86)', 'ProgramFiles'):
        if os.environ.get(env):
            steam_roots.append(Path(os.environ[env])/'Steam')
    if os.name == 'nt':
        import winreg
        for hive, key, field in ((winreg.HKEY_CURRENT_USER, r'Software\Valve\Steam', 'SteamPath'),
                                 (winreg.HKEY_LOCAL_MACHINE, r'Software\Valve\Steam', 'InstallPath')):
            for view in (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
                try:
                    with winreg.OpenKey(hive, key, 0, winreg.KEY_READ | view) as handle:
                        steam_roots.append(Path(winreg.QueryValueEx(handle, field)[0]))
                except OSError:
                    pass
    for steam in dict.fromkeys(steam_roots):
        for library in steam_libraries(steam):
            candidates.append(library/'steamapps/common/Detroit Become Human/BigFile_PC.idx')
    if os.environ.get('ProgramFiles'):
        candidates.append(Path(os.environ['ProgramFiles'])/'Epic Games/DetroitBecomeHuman/BigFile_PC.idx')
    if os.environ.get('ProgramData'):
        directory = Path(os.environ['ProgramData'])/'Epic/EpicGamesLauncher/Data/Manifests'
        try:
            manifests = list(directory.glob('*.item'))
        except OSError:
            manifests = []
        for file in manifests:
            try:
                item = json.loads(file.read_text(encoding='utf-8-sig'))
                location = item.get('InstallLocation')
                if location:
                    candidates.append(Path(location)/'BigFile_PC.idx')
            except (OSError, ValueError, AttributeError):
                pass
    found = {}
    for path in candidates:
        if path.is_file():
            found.setdefault(os.path.normcase(str(path.resolve())), path)
    return list(found.values())


def choose_index(configured='', saved='', candidates=None, abspath=None):
    selected = normalize_index(configured, abspath)
    if selected is not None:
        return selected  # Even if missing: report THIS path, not another edition.
    previous = normalize_index(saved, abspath)
    if previous is not None and previous.is_file():
        return previous
    available = installed_indexes() if candidates is None else list(candidates)
    return Path(available[0]) if len(available) == 1 else None


def resolve_index(context=None, metadata=None, preferences=None, required=False):
    import bpy
    context = context or bpy.context
    if preferences is None:
        addons = getattr(getattr(context, 'preferences', None), 'addons', None)
        addon = addons.get(__package__) if addons is not None else None
        preferences = addon.preferences if addon is not None else None
    path = choose_index(getattr(preferences, 'game_index', ''),
                        (metadata or {}).get('game_index', ''), abspath=bpy.path.abspath)
    if required:
        if path is None:
            raise ValueError('Select your Steam or Epic BigFile_PC.idx in the Detroit add-on preferences')
        if not path.is_file():
            raise FileNotFoundError(f'Selected BigFile_PC.idx not found: {path}. No default-path fallback was used.')
    return path
