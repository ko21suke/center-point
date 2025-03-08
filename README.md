# TODO after building point encoder and beckbone
* config fileの修正
* config fileのmodelのパラメータ名の修正
* tools/solver/fastai_optim.py の FIXMEの修正
* tools/torchie/trainer/trainer.py # 358のFIXMEを修正: device=を使用しないようにコメントアウトした
* tools/core/bbox/geometry.py　のpoints_in_convex_polygon_jitのデコレーターの()を追加した
* Center Point の self.modulesの実装 [nn.Sequential, nn.ModuleList, self.add_modules()どれかを使用]