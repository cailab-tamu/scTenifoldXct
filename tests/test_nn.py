import pytest


@pytest.fixture(scope="session")
def test_nn(xct_test):
    emb = xct_test.get_embeds(train=True)
    return xct_test.null_test()