-- Symphony by vyrx
-- Theme: Gruvbox Material
-- https://github.com/vyrx-dev

return {
	{
		"sainnhe/gruvbox-material",
		lazy = true,
		priority = 1000,
		opts = {},
		config = function()
			vim.g.gruvbox_material_background = "hard"
			vim.g.gruvbox_material_transparent_background = 1
		end,
	},
	{
		"LazyVim/LazyVim",
		opts = {
			colorscheme = "gruvbox-material",
		},
	},
}
