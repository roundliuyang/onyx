import React from "react";
import { render, screen, setupUser } from "@tests/setup/test-utils";
import SimpleModelSelector from "@/lib/modelSelectors/components/SimpleModelSelector";
import type { ModelOptionProvider } from "@/lib/languageModels/types";

// The listbox renders through a portal; keep it inside the test container.
jest.mock("react-dom", () => ({
  ...jest.requireActual("react-dom"),
  createPortal: (node: React.ReactNode) => node,
}));
Element.prototype.scrollIntoView = jest.fn();

const providers: ModelOptionProvider[] = [
  {
    id: 1,
    name: "OpenAI",
    provider: "openai",
    model_configurations: [
      {
        id: 11,
        name: "gpt-4o",
        is_visible: true,
        max_input_tokens: null,
        supports_image_input: true,
        supports_reasoning: false,
        effectiveDisplayName: "GPT-4o",
      },
      {
        id: 12,
        name: "gpt-4.1",
        is_visible: true,
        max_input_tokens: null,
        supports_image_input: false,
        supports_reasoning: false,
        effectiveDisplayName: "GPT-4.1",
      },
    ],
  },
];

describe("SimpleModelSelector", () => {
  test("nullable: nothing chosen shows the placeholder, never a fallback model", () => {
    render(
      <SimpleModelSelector
        nullable
        providers={providers}
        value={null}
        onChange={jest.fn()}
      />
    );
    expect(screen.getByRole("combobox", { name: "Select model" })).toHaveValue(
      ""
    );
  });

  test("shows the chosen model and emits the picked id", async () => {
    const handleChange = jest.fn();
    const user = setupUser();
    render(
      <SimpleModelSelector
        providers={providers}
        value={11}
        onChange={handleChange}
      />
    );
    expect(screen.getByRole("combobox")).toHaveValue("GPT-4o");

    await user.click(screen.getByRole("combobox"));
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    await user.click(screen.getByRole("option", { name: /GPT-4\.1/ }));
    expect(handleChange).toHaveBeenCalledWith(12);
  });

  test("non-nullable: re-picking the chosen model changes nothing", async () => {
    const handleChange = jest.fn();
    const user = setupUser();
    render(
      <SimpleModelSelector
        providers={providers}
        value={11}
        onChange={handleChange}
      />
    );
    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByRole("option", { name: /GPT-4o/ }));
    expect(handleChange).not.toHaveBeenCalled();
    expect(screen.getByRole("combobox")).toHaveValue("GPT-4o");
  });

  test("nullable: re-picking the chosen model clears it", async () => {
    const handleChange = jest.fn();
    const user = setupUser();
    render(
      <SimpleModelSelector
        nullable
        providers={providers}
        value={11}
        onChange={handleChange}
      />
    );
    await user.click(screen.getByRole("combobox"));
    await user.click(screen.getByRole("option", { name: /GPT-4o/ }));
    expect(handleChange).toHaveBeenCalledWith(null);
  });

  test("renders an empty list without options", async () => {
    const user = setupUser();
    render(
      <SimpleModelSelector
        nullable
        providers={[]}
        value={null}
        onChange={jest.fn()}
      />
    );
    await user.click(screen.getByRole("combobox"));
    expect(screen.queryByRole("option")).not.toBeInTheDocument();
  });
});
