"""Import-level smoke test for the vispy 3D scene module.

Constructing a full Scene3D requires a GL context, which is not available under
the offscreen Qt platform used by the test suite. Here we only verify the module
and class are importable and expose the expected structure.
"""


def test_scene3d_module_importable():
    import app.scene3d as s3

    assert hasattr(s3, "Scene3D")
    assert s3.scene  # vispy scene namespace accessible
    assert s3.visuals  # visual classes accessible
