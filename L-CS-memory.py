'Run the existing ImageNet-C evaluator with a Table-4 memory control.'
import os
import runpy
from pathlib import Path
import methods.ECHO
from methods.ECHO.simple_memories import replace_memory
wrap_echo = methods.ECHO.wrap_adapter

def wrap_memory(adapter, cfg, source_model=None):
    if os.environ['ECHO_MEMORY_KIND'] == 'plain-slot-independent':
        from methods.slot_memory import SlotMemory
        if cfg.ADAPTER.NAME != 'Tent':
            raise ValueError('Independent slot memory is configured for Tent only')
        return SlotMemory(adapter, source_model, cfg.ADAPTER.ECHO.FEATURES, cfg.ADAPTER.ECHO.DIAGNOSTIC_PATH)
    if os.environ['ECHO_MEMORY_KIND'] == 'plain-online-independent':
        from methods.online_clustering import OnlineClustering
        if cfg.ADAPTER.NAME != 'Tent':
            raise ValueError('Independent online clustering is configured for Tent only')
        return OnlineClustering(adapter, source_model, cfg.ADAPTER.ECHO.DIAGNOSTIC_PATH)
    wrapper = wrap_echo(adapter, cfg, source_model)
    return replace_memory(wrapper, os.environ['ECHO_MEMORY_KIND'], int(os.environ.get('ECHO_MEMORY_CLUSTERS', '15')), float(os.environ.get('ECHO_MEMORY_THRESHOLD', '1')), source_model=source_model, temperature=float(os.environ.get('ECHO_MEMORY_TEMPERATURE', '1')))
if __name__ == '__main__':
    methods.ECHO.wrap_adapter = wrap_memory
    runpy.run_path(str(Path(__file__).with_name('L-CS.py')), run_name='__main__')
