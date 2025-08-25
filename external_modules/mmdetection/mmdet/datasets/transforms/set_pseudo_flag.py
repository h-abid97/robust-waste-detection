from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

@TRANSFORMS.register_module()
class SetPseudoFlag(BaseTransform):
    def __init__(self, flag_value=False):
        super().__init__()
        self.flag_value = flag_value

    def transform(self, results):
        # Set in multiple locations to ensure it persists
        results['is_pseudo'] = self.flag_value
        
        # Set in metainfo
        if 'metainfo' not in results:
            results['metainfo'] = {}
        results['metainfo']['is_pseudo'] = self.flag_value
        
        # Also set in img_meta for backward compatibility
        if 'img_meta' not in results:
            results['img_meta'] = {}
        results['img_meta']['is_pseudo'] = self.flag_value
        
        #print(f"DEBUG: SetPseudoFlag - results keys: {results.keys()}")
        #print(f"DEBUG: metainfo: {results.get('metainfo', {})}")
        #print(f"DEBUG: img_meta: {results.get('img_meta', {})}")
        
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(flag_value={self.flag_value})"